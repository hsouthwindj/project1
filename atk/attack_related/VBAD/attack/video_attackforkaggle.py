import collections
import logging
import numpy as np
import torch
# from attack.group_generator import EquallySplitGrouping
from ..attack.group_generator import EquallySplitGrouping
from torch import nn
import torchvision

from network.classifier import Meso4

from model.classifiers import DeepFakeClassifier

import re

import sys
sys.path.append('../../..')

#image_model_path = '/kaggle/input/models/all_c40.p'

device = torch.device('cuda')

def video_score(img_model, vid):
    with torch.no_grad():
        re = 0
        for i in range(len(vid)):
            re += img_model(vid[i].unsqueeze(0))[0][1]
    return re
    
def video_score_meso(img_model, vid):            
    with torch.no_grad():
        rsf = torchvision.transforms.Resize((256, 256))
        vid = rsf(vid)                             
        re = 0
        for i in range(len(vid)):
            print(img_model(vid[i].unsqueeze(0)))
            re += img_model(vid[i].unsqueeze(0))[0][0]
    return re

def fake_rate(img_model, vid):
    with torch.no_grad():
        re = 0
        for i in range(len(vid)):
            out = img_model(vid[i].unsqueeze(0))
            if out[0][1] > out[0][0]:
                re += 1
    return re / len(vid)

def fake_rate_meso(img_model, vid):
    with torch.no_grad():
        re = 0
        rsf = torchvision.transforms.Resize((256, 256))
        vid = rsf(vid)
        for i in range(len(vid)):
            out = img_model(vid[i].unsqueeze(0))
            if out[0][1] < out[0][0]:
                re += 1
    return re / len(vid)

def img_pert(img_model, img):
    modif = torch.rand(img.shape).to(device) * (1/255)
    pert = torch.nn.Parameter(modif)
    optimizer = torch.optim.Adam([pert], lr=0.01)
    fn = torch.nn.Softmax(dim = 1)
    x = img + pert
    it = 0
    x = x.unsqueeze(0)
    while it < 100:
        it += 1
        out = img_model(x)
        out = fn(out)
        
        if out[0][1] < 0.25:
            break
            
        loss = -torch.log(1 - out[0][1])

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    # print('cur score', fake_score, 'it', it, 'sp_rate', sp_rate)
    return pert
    
def predict_image(model, img, model_type):
    # preprocess ??
    # print(img.shape)
    f = nn.Softmax(dim = 1)
    if model_type == 'xception':
        out = model(img.unsqueeze(dim = 0))
        out = f(out)
        # print('img prediction ', out)
        _, prediction = torch.max(out, dim = 1)
        # print(prediction)

        prediction = float(prediction.cpu().numpy()) # prediction == 1 => fake

        return int(prediction), out
    elif model_type == 'meso':
        rsf = torchvision.transforms.Resize((256, 256))
        imgs = rsf(img)
        out = model(imgs.unsqueeze(0))
        out = f(out)
        # print('prediction', out)
        _, prediction = torch.max(out, dim = 1)
        # print(prediction)
        prediction = float(prediction.cpu().numpy())

        return int(prediction), out

def image_checker(X, model, model_type): # X dim = 5
    X = X.unsqueeze(0)
    f = 0
    if model_type == 'meso':
        for i in range(len(X[0])):
            t, _ = predict_image(model, X[0][i], 'meso')
            if t == 0:
                f += 1
    elif model_type == 'xception':
        for i in range(len(X[0])):
            t, _ = predict_image(model, X[0][i], 'xception')
            f += t
    elif model_type == 'ef':
        with torch.no_grad():
            rsf = torchvision.transforms.Resize(380)
            for i in range(len(X[0])):
                t = model(rsf(X[0][i].unsqueeze(0)))
                if t[0] < 0:
                    f += 1
    print('detector %s, origin total frame %d, origin total fake frame %d', model_type, len(X[0]), f)    
        

def sim_rectification_vector(model, vid, tentative_directions, n, sigma, target_class, rank_transform, sub_num,
                             group_gen, img_model, fake_r, image_flag, untargeted, vc = 'rnn'):
    with torch.no_grad():
        image_flag = False
        sigma *= (100*fake_r + 1)
        #model.cuda()
        grads = torch.zeros(len(group_gen), device='cuda')
        count_in = 0
        loss_total = 0
        logging.info('sampling....')
        batch_loss = []
        batch_noise = []
        batch_idx = []
        # print(n, sub_num)
        assert n % sub_num == 0 and sub_num % 2 == 0
        for _ in range(n // sub_num):
            adv_vid_rs = vid.repeat((sub_num,) + (1,) * len(vid.size())).cuda() # [sub_num, frame_num, c, w, h]
            
            # noise_list = torch.randn((sub_num // 2,) + grads.size(), device='cuda') * sigma # [sub_num // 2, len(group_gen)]
             
            
            # own modification
            # only take the better result for the image detector
            noise_list = torch.randn((int(sub_num // 2),) + grads.size(), device='cuda') * sigma
            
            # after find the proper noise for image detector, filter those gave negative impact for video detector
            # impact is prob1 - prob2, which are the result of vid_model(X + noise), vid_model(X - noise)
            
            
            # change noise list and adv_vid_rs for model input size compatiable
            # adv_vid_rs = vid.clone()
            # noise_list = torch.randn(grads.size(), device='cuda') * sigma
            all_noise = torch.cat([noise_list, -noise_list], 0)

            perturbation_sample = group_gen.apply_group_change(tentative_directions, all_noise) # [frame_num, c, w, h], [sub_num, len(group_gen)]
            #if image_flag:
            #    ns = [(video_score(img_model, (vid.repeat((len(all_noise),) + (1,) * len(vid.size())).cuda() + perturbation_sample)[i]), perturbation_sample[i], all_noise[i]) for i in range(len(all_noise))]
            if image_flag:
                ns = [(video_score_meso(img_model, (vid.repeat((len(all_noise),) + (1,) * len(vid.size())).cuda() + perturbation_sample)[i]), perturbation_sample[i], all_noise[i]) for i in range(len(all_noise))]
            # ns = sorted(ns, key = lambda x : x[0])
            # all_noise = torch.stack([ns[i][2] for i in range(sub_num)])
            # perturbation_sample = torch.stack([ns[i][1] for i in range(sub_num)])
            
            adv_vid_rs += perturbation_sample
            del perturbation_sample
            
            # top_val, top_idx, logits = model(adv_vid_rs)
            # temp = [model(adv_vid_rs[i].unsqueeze(0)) for i in range(sub_num)]
            if vc == 'rnn':
                logits, cnn_y = model(adv_vid_rs)
            else:
                logits = torch.tensor([[model.get_score(model.ori_classify(adv_vid_rs[i]))] for i in range(sub_num)]).cuda()
            top_idx = torch.sigmoid(logits) > 0.5
            if image_flag:
                loss = -torch.max(logits, 1)[0] - torch.stack([ns[i][0] for i in range(len(ns))]) * fake_r
            else:
                loss = -torch.max(logits, 1)[0]
            
#             print(-torch.max(logits, 1)[0])
#             print(torch.stack([ns[i][0] for i in range(len(ns))]))

            batch_loss.append(loss)
            batch_idx.append(top_idx.long()) # need to convert bool to int
            # batch_noise.append(all_noise)
            batch_noise.append(all_noise)
        batch_noise = torch.cat(batch_noise, 0)
        batch_idx = torch.cat(batch_idx)
        batch_loss = torch.cat(batch_loss)

        # Apply rank-based loss transformation
        if rank_transform:
            good_idx = torch.sum(batch_idx == target_class, 1).byte()
            changed_loss = torch.where(good_idx, batch_loss, torch.tensor(1000., device='cuda'))
            loss_order = torch.zeros(changed_loss.size(0), device='cuda')
            sort_index = changed_loss.sort()[1]
            loss_order[sort_index] = torch.arange(0, changed_loss.size(0), device='cuda', dtype=torch.float)
            available_number = torch.sum(good_idx).item()
            count_in += available_number
            unavailable_number = n - available_number
            unavailable_weight = torch.sum(torch.where(good_idx, torch.tensor(0., device='cuda'),
                                                       loss_order)) / unavailable_number if unavailable_number else torch.tensor(
                0., device='cuda')
            rank_weight = torch.where(good_idx, loss_order, unavailable_weight) / (n - 1)
            grads += torch.sum(batch_noise / sigma * (rank_weight.view((-1,) + (1,) * (len(batch_noise.size()) - 1))),
                               0)
        else:
            idxs = (batch_idx == target_class).nonzero()
            valid_idxs = idxs[:, 0].cuda()
            valid_loss = torch.index_select(batch_loss, 0, valid_idxs)
            
            loss_total += torch.mean(valid_loss).item()
            count_in += valid_loss.size(0)
            noise_select = torch.index_select(batch_noise, 0, valid_idxs)
            # a = torch.sum(noise_select / sigma * (valid_loss.view((-1,) + (1,) * (len(noise_select.size()) - 1))),
                               # 0)
            
            grads += torch.sum(noise_select / sigma * (valid_loss.view((-1,) + (1,) * (len(noise_select.size()) - 1))),
                               0)

        if count_in == 0:
            return None, None
        logging.info('count in: {}'.format(count_in))
        return loss_total / count_in, grads


# Input video should be torch.tensor and its shape should be [num_frames, c, w, h]
# The input should be normalized to [0, 1]
def untargeted_video_attack(vid_model, vid, directions_generator, ori_class,
                            rank_transform=False, eps=0.05, max_lr=1e-2, min_lr=2 * 1e-3, sample_per_draw=48,
                            max_iter=10000, sigma=1e-5, sub_num_sample=12, image_split=1, vc='rnn', img_model=None):
    num_iter = 0
    adv_vid = torch.clamp(vid.clone() + (torch.rand_like(vid) * 2 - 1) * eps, 0., 1.).cuda()
    cur_lr = max_lr
    last_p = []
    last_score = []
    group_gen = EquallySplitGrouping(image_split)
    
    #img_model = torch.load(image_model_path)
    max_lr = 2*1e-2
    min_lr = 1*1e-3
    
    fake_rate_mi = 0.25
    fake_rate_ma = 0.5
    image_flag = True
    
    #meso img model
    #cp = torch.load('/kaggle/input/othermodels/best.pkl')
    #mesomodel = Meso4()
    #mesomodel.load_state_dict(cp)
    #mesomodel = mesomodel.cuda()
    
    #pth = '/kaggle/input/othermodels/final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36'
    #efmodel = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns").to("cuda")
    #checkpoint = torch.load(pth, map_location="cpu")
    #state_dict = checkpoint.get("state_dict", checkpoint)
    #efmodel.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=True)
    #efmodel.eval()

    while num_iter < max_iter:
        #ip = img_pert(img_model, adv_vid[len(adv_vid) // 2].detach().clone())
        #with torch.no_grad():
        #    adv_vid[len(adv_vid) // 2] += ip
        #clip_frame = torch.clamp(adv_vid, 0., 1.)
        #adv_vid = clip_frame.clone()
        #fake_r = fake_rate(img_model, adv_vid)
        #fake_r = fake_rate_meso(mesomodel, adv_vid)
        #if fake_r < fake_rate_mi:
        #    image_flag = False
        #elif fake_r > fake_rate_ma:
        #    image_flag = True
        #print('image fake rate', fake_r)
        fake_r = 1
        image_flag = False
        # print(adv_vid.shape)
        # print(adv_vid[None, :].shape)
        # top_val, top_idx, _ = vid_model(adv_vid[None, :])
        with torch.no_grad():
            if vc != 'rnn':
                top_val = vid_model.get_score(vid_model.ori_classify(adv_vid))
                top_val = torch.tensor([[top_val, 1 - top_val]]) # [fake, real]
                top_idx = 1 if top_val[0][0] >= 0.5 else 0
            else:
                top_val = torch.sigmoid(vid_model(adv_vid.unsqueeze(dim = 0))[0])
                top_idx = 1 if top_val.mean() > 0.5 else 0
        num_iter += 1
        if ori_class != top_idx and image_flag == False:
            print('early stop', num_iter)
            image_checker(adv_vid, img_model, 'xception')
            #image_checker(adv_vid, mesomodel, 'meso')
            #image_checker(adv_vid, efmodel, 'ef')
            logging.info('early stop at iterartion {}'.format(num_iter))
            return True, num_iter, adv_vid
        idx = [[0, 0]] # idk waht's this but it's always [0, 0] when i testing
        pre_score = top_val[0][idx[0][1]]
        logging.info('cur target prediction: {}'.format(pre_score))
        print('cur target prediction: {}'.format(pre_score))
        print('cur prediction label: {}'.format(top_idx))
        # print(top_val, pre_score, idx)
        # tensor([[0.9987]], device='cuda:0') tensor(0.9987, device='cuda:0') tensor([[0, 0]], device='cuda:0')

        last_score.append(pre_score)
        last_score = last_score[-200:]
        if last_score[-1] >= last_score[0] and len(last_score) == 200:
            image_checker(adv_vid, img_model, 'xception')
            #image_checker(adv_vid, mesomodel, 'meso')
            #image_checker(adv_vid, efmodel, 'ef')
            print('FAIL: No Descent, Stop iteration')
            return False, pre_score.cpu().item(), adv_vid

        # Annealing max learning rate
        last_p.append(pre_score)
        last_p = last_p[-20:]
        if last_p[-1] <= last_p[0] and len(last_p) == 20:
            if cur_lr > min_lr:
                print("[log] Annealing max_lr")
                cur_lr = max(cur_lr / 2., min_lr)
            last_p = []
        
        tentative_directions = directions_generator(adv_vid).cuda()
        group_gen.initialize(tentative_directions)
        
        l, g = sim_rectification_vector(vid_model, adv_vid, tentative_directions, sample_per_draw, sigma,
                                        ori_class, rank_transform, sub_num_sample, group_gen, mesomodel, fake_r, image_flag, untargeted=True,vc=vc)

        if l is None and g is None:
            logging.info('nes sim fails, try again....')
            continue

        # Rectify tentative perturabtions
        assert g.size(0) == len(group_gen), 'rectification vector size error!'
        rectified_directions = group_gen.apply_group_change(tentative_directions, torch.sign(g))
        num_iter += sample_per_draw

        proposed_adv_vid = adv_vid

        assert proposed_adv_vid.size() == rectified_directions.size(), 'rectification error!'
        # PGD
        proposed_adv_vid += cur_lr * rectified_directions
        bottom_bounded_adv = torch.where((vid - eps) > proposed_adv_vid, vid - eps,
                                         proposed_adv_vid)
        bounded_adv = torch.where((vid + eps) < bottom_bounded_adv, vid + eps, bottom_bounded_adv)
        clip_frame = torch.clamp(bounded_adv, 0., 1.)
        adv_vid = clip_frame.clone()

        logging.info('step {} : loss {} | lr {}'.format(num_iter, l, cur_lr))
    image_checker(adv_vid, img_model, 'xception')
    #image_checker(adv_vid, mesomodel, 'meso')
    #image_checker(adv_vid, efmodel, 'ef')
    return False, pre_score.cpu().item(), adv_vid
