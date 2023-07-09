'''
author cg563
source : https://github.com/cg563/simple-blackbox-attack/blob/master/simba.py
'''

import torch
import torch.nn.functional as F
import utils
import torchvision


class SimBA:
    
    def __init__(self, model, dataset, image_size):
        self.model = model
        self.dataset = dataset
        self.image_size = image_size
        self.model.eval()
        
    def expand_vector(self, x, size):
        batch_size = x.size(0)
        x = x.view(-1, 3, size, size)
        z = torch.zeros(batch_size, 3, self.image_size, self.image_size)
        z[:, :, :size, :size] = x
        return z

    def get_probs(self, x, y):
        # print(x.shape)
        # for 3dcnn
        brsf = torchvision.transforms.Resize((224, 192))
        x = brsf(x)
        output = self.model((x.cuda()))
        # output = torch.sigmoid(output)
        probs = torch.index_select(F.softmax(output, dim=-1).data, 1, y)
        return torch.diag(probs)
    
    def get_preds(self, x):
        # for 3dcnn
        brsf = torchvision.transforms.Resize((224, 192))
        x = brsf(x)
        output = self.model(x.cuda()).cpu()
        _, preds = output.data.max(1)
        return preds

    # 20-line implementation of SimBA for single image input
    def simba_single(self, x, y, num_iters=10000, epsilon=0.5 / 255, targeted=False):
        n_dims = x.view(1, -1).size(1)
        perm = torch.randperm(n_dims)
        x = x.unsqueeze(0)
        last_prob = self.get_probs(x, y)
        for i in range(num_iters):
            diff = torch.zeros(n_dims).cuda()
            diff[perm[i]] = epsilon
            left_prob = self.get_probs((x - diff.view(x.size())).clamp(0, 1), y)
            if targeted != (left_prob < last_prob):
                x = (x - diff.view(x.size())).clamp(0, 1)
                last_prob = left_prob
            else:
                right_prob = self.get_probs((x + diff.view(x.size())).clamp(0, 1), y)
                if targeted != (right_prob < last_prob):
                    x = (x + diff.view(x.size())).clamp(0, 1)
                    last_prob = right_prob
            if i % 10 == 0:
                print(left_prob, right_prob)
                print(last_prob)
        return x.squeeze()
    def simba_batch(self, images_batch, labels_batch, max_iters, freq_dims, stride, epsilon, linf_bound=0.0,
                order='rand', targeted=False, pixel_attack=False, log_every=10):
        batch_size = images_batch.size(0)
        image_size = images_batch.size(2)
        assert self.image_size == image_size
        # sample a random ordering for coordinates independently per batch element
        if order == 'rand':
            # indices = torch.randperm(3 * freq_dims * freq_dims)[:max_iters]
            indices = torch.randperm(3 * freq_dims * freq_dims)#[:max_iters]
        elif order == 'diag':
            indices = utils.diagonal_order(image_size, 3)[:max_iters]
        elif order == 'strided':
            indices = utils.block_order(image_size, 3, initial_size=freq_dims, stride=stride)[:max_iters]
        else:
            indices = utils.block_order(image_size, 3)[:max_iters]
        if order == 'rand':
            expand_dims = freq_dims
        else:
            expand_dims = image_size
        n_dims = 3 * expand_dims * expand_dims
        x = torch.zeros(batch_size, n_dims)#.fill_(2/255)
        # logging tensors
        probs = torch.zeros(batch_size, max_iters).cuda()
        succs = torch.zeros(batch_size, max_iters).cuda()
        queries = torch.zeros(batch_size, max_iters).cuda()
        l2_norms = torch.zeros(batch_size, max_iters).cuda()
        linf_norms = torch.zeros(batch_size, max_iters).cuda()
        prev_probs = self.get_probs(images_batch, labels_batch).cuda()
        preds = self.get_preds(images_batch).cuda()
        if pixel_attack:
            trans = lambda z: z
        else:
            trans = lambda z: utils.block_idct(z, block_size=image_size, linf_bound=linf_bound)
        remaining_indices = torch.arange(0, batch_size).long()
        for k in range(max_iters):
            dim = indices[k % len(indices)]
            expanded = (images_batch[remaining_indices] + trans(self.expand_vector(x[remaining_indices], expand_dims)).cuda()).clamp(0, 1)
            perturbation = trans(self.expand_vector(x, expand_dims)).cuda()
            l21 = torch.sum(torch.sqrt(torch.mean(torch.pow((perturbation).unsqueeze(0), 2), dim=0).mean(dim=2).mean(dim=2).mean(dim=1)))
            #print(l21)
            l2_norms[:, k] = perturbation.view(batch_size, -1).norm(2, 1)
            linf_norms[:, k] = perturbation.view(batch_size, -1).abs().max(1)[0]
            preds_next = self.get_preds(expanded).cuda()
            preds[remaining_indices] = preds_next
            if k == 0:
                remaining = preds.eq(labels_batch)
            if targeted:
                remaining = preds.ne(labels_batch)
            else:
                remaining = preds.eq(labels_batch)
            # check if all images are misclassified and stop early
            if remaining.sum() == 0:
                adv = (images_batch + trans(self.expand_vector(x, expand_dims)).cuda()).clamp(0, 1)
                probs_k = self.get_probs(adv, labels_batch).cuda()
                probs[:, k:] = probs_k.unsqueeze(1).repeat(1, max_iters - k)
                succs[:, k:] = torch.ones(batch_size, max_iters - k)
                queries[:, k:] = torch.zeros(batch_size, max_iters - k)
                break
            remaining_indices = torch.arange(0, batch_size)[remaining.cpu()].long()
            if k > 0:
                succs[:, k] = ~remaining
            diff = torch.zeros(remaining.sum(), n_dims)
            #diff[:, k] = epsilon
            diff[:, torch.randint(0, len(indices), (32,))] = epsilon
            diff[:, torch.randint(0, len(indices), (32,))] = -epsilon
            left_vec = x[remaining_indices] - diff
            right_vec = x[remaining_indices] + diff
            # trying negative direction
            adv = (images_batch[remaining_indices] + trans(self.expand_vector(left_vec, expand_dims)).cuda()).clamp(0, 1)
            left_probs = self.get_probs(adv, labels_batch[remaining_indices])
            queries_k = torch.zeros(batch_size)
            # increase query count for all images
            queries_k[remaining_indices] += 1
            if targeted:
                improved = left_probs.gt(prev_probs[remaining_indices])
            else:
                improved = left_probs.lt(prev_probs[remaining_indices])
            # only increase query count further by 1 for images that did not improve in adversarial loss
            if improved.sum() < remaining_indices.size(0):
                queries_k[remaining_indices[~improved.cpu()].cpu()] += 1
            # try positive directions
            adv = (images_batch[remaining_indices] + trans(self.expand_vector(right_vec, expand_dims)).cuda()).clamp(0, 1)
            right_probs = self.get_probs(adv, labels_batch[remaining_indices])
            if targeted:
                right_improved = right_probs.gt(torch.max(prev_probs[remaining_indices], left_probs))
            else:
                right_improved = right_probs.lt(torch.min(prev_probs[remaining_indices], left_probs))
            probs_k = prev_probs.clone()
            # update x depending on which direction improved
            if improved.sum() > 0:
                left_indices = remaining_indices[improved.cpu()]
                left_mask_remaining = improved.unsqueeze(1).repeat(1, n_dims)
                x[left_indices] = left_vec[left_mask_remaining.cpu()].view(-1, n_dims)
                probs_k[left_indices] = left_probs[improved]
            if right_improved.sum() > 0:
                right_indices = remaining_indices[right_improved.cpu()]
                right_mask_remaining = right_improved.unsqueeze(1).repeat(1, n_dims)
                x[right_indices] = right_vec[right_mask_remaining.cpu()].view(-1, n_dims)
                probs_k[right_indices] = right_probs[right_improved]
            probs[:, k] = probs_k
            queries[:, k] = queries_k
            prev_probs = probs[:, k]
            if (k + 1) % log_every == 0 or k == max_iters - 1:
                print('Iteration %d: queries = %.4f, prob = %.4f, remaining = %.4f' % (
                        k + 1, queries.sum(1).mean(), probs[:, k].mean(), remaining.float().mean()))
            # if probs[:, k].mean() < 0.1:
            #     break
        expanded = (images_batch + trans(self.expand_vector(x, expand_dims)).cuda()).clamp(0, 1)
        preds = self.get_preds(expanded).cuda()
        if targeted:
            remaining = preds.ne(labels_batch)
        else:
            remaining = preds.eq(labels_batch.cuda())
        succs[:, max_iters-1] = ~remaining
        print('inner', self.model(expanded)[:10])
        return expanded, probs, succs, queries, l2_norms, linf_norms