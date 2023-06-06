import os
import yaml
import glob

import numpy as np
import cv2

import torch
from torch import nn, autograd
from torch.utils.data import Dataset, DataLoader

from torchvision.models.detection.transform import GeneralizedRCNNTransform

from albumentations import Compose, SmallestMaxSize, CenterCrop, Normalize, PadIfNeeded
from albumentations.pytorch.transforms import ToTensor

from dsfacedetector.face_ssd_infer import SSD
from tracker.iou_tracker import track_iou
from efficientnet_pytorch.model import EfficientNet, MBConvBlock

from torchvision import transforms
import torchvision
import cv2
from facenet_pytorch.models.mtcnn import MTCNN
from utils.aucloss import AUCLoss
from sklearn.metrics import accuracy_score

import detector
import GAfunc
import logging
import time

from scipy import spatial
from torchsummary import summary
from torchmetrics import StructuralSimilarityIndexMeasure

mid_out = []

def hook_fn(module, i, o): #module, input, output
    mid_out.append(o)

def group_diff(vs):
    return sum([torch.pow((vs[i + 1] - vs[i]), 2).flatten().sum() for i in range(len(vs) - 1)])

def group_cs_center(vs, c):
    if len(vs) <= c:
        return group_cs(vs)
    re = []
    for i in range(len(vs)):
        re.append(-1 * (spatial.distance.cosine(vs[c].flatten().detach().cpu(), vs[i].flatten().detach().cpu()) - 1))
    return sum(re)


def group_cs(vs):
    # print(vs.shape)
    total = sum(vs)
    return sum([-1 * (spatial.distance.cosine(total.flatten().detach().cpu(), vs[i].flatten().detach().cpu()) - 1) for i in range(len(vs))])
    # return sum([-1 * (spatial.distance.cosine(vs[i].flatten().detach().cpu(), vs[i + 1].flatten().detach().cpu()) - 1) for i in range(len(vs) - 1)])

def group_img_dec(vs, model, model_type):
    fake = 0
    for img in vs:
        if predict_image(model, img, model_type)[0] == 1:
            fake += 1 # detect fake means spoof failed
    return fake / len(vs) # range 0 ~ 1

def video_loader(url):
    with open('/notebooks/Deepfake-detection-master/rnnpass') as file:
        ok = [line.rstrip()[10:] for line in file]
    if os.path.isdir(url):
        vids = []
        for vid_name in os.listdir(url):
            vid_path = os.path.join(url, vid_name)
            if vid_path not in ok:
                continue
            try:
                vids.append((vid_path, detector.load_data(vid_path, device)))
            except:
                pass
    else:
        vids = [(url, detector.load_data(url, device))]
    print(len(vids))
    return vids

def predict_image(model, img, model_type):
    # preprocess ??
    # print(img.shape)
    f = nn.Softmax(dim = 1)
    if model_type == 'xception':
        out = model(img.unsqueeze(dim = 0))
    out = f(out)
    _, prediction = torch.max(out, dim = 1)
    # print(prediction)
    
    prediction = float(prediction.cpu().numpy()) # prediction == 1 => fake
    
    return int(prediction), out

def image_pert(model, ori_imgs, perts, model_type, target_frame = -1, group_size = -1):
    perts.requires_grad = True
    optimizer = torch.optim.Adam([perts], lr=0.01) # 0.0002 too low, 0.001 ok , 0.0004 almost there
    # print(perts.shape)
    fn = nn.Softmax(dim = 1)
    it = 0
    
    
    while it < 100:
        input_var = ori_imgs + perts # need to put inside the loop, or the fake score will be the same for some reason
        it += 1
        if model_type == 'xception':
            out = model(input_var[0])
        out = fn(out)
        if target_frame != -1 and group_size != -1:
            tarr = [out[i][1] for i in range(len(out)) if i % group_size == target_frame]
            fake_score = sum(tarr) / len(tarr)
            if fake_score < 0.5:
                break
        else:
            sp_rate = sum([out[i][1] < out[i][0] for i in range(len(out))]) / len(out)
            fake_score = sum([out[i][1] for i in range(len(out))]) / len(out)
            if sp_rate > 0.9:
                break
            
        loss = -torch.log(1 - fake_score)

        loss.backward()
        optimizer.step()
    return perts

def print_output(model, img_model, vid, group_size, model_type):
    
    probs, pre_label = model(vid)
    probs = torch.sigmoid(probs)
    print('vid output', probs)
    
    f = 0
    fa = [0] * group_size
    for i in range(len(vid[0])):
        t, _ = predict_image(img_model, vid[0][i], model_type)
        f += t
        fa[i % group_size] += t
    print('total frame %d, total fake frame %d', len(vid[0]), f)
    print('fake array', fa)
    
def test_func(img_model, model, X):
    summary(img_model, X[0].shape[1:])
    summary(model, X.shape[1:])
    print(vars(img_model))
    print('=' * 100)
    print(vars(model))
    
    global mid_out
    mid_out = []
    # 2048, 2, 2
    # img_model.model.conv4.register_forward_hook(hook_fn)
    img_model.model.bn4.register_forward_hook(hook_fn)
    _ = img_model(X[0])
    img_mid = [i for i in mid_out[0]]
    img_mid = torch.cat(img_mid)
    
    for i in range(3):
        print(X.shape)
        mid_out = []
        thk = model.module.cnn[7][i].conv3.register_forward_hook(hook_fn)
        _ = model(X)
        tmid = []
        for j in mid_out:
            if j.shape == [1, 2048, 2, 2]:
                tmid.append(j)
        mid_out = torch.cat(mid_out)
        cs = (-1 * (spatial.distance.cosine(mid_out.flatten().detach().cpu(), img_mid.flatten().detach().cpu()) - 1))
        print(i, 'th seq conv3 cs is ', cs)
        thk.remove()
        
        mid_out = []
        thk = model.module.cnn[7][i].bn3.register_forward_hook(hook_fn)
        _ = model(X)
        mid_out = torch.cat(mid_out)
        cs = (-1 * (spatial.distance.cosine(mid_out.flatten().detach().cpu(), img_mid.flatten().detach().cpu()) - 1))
        print(i, 'th seq bn3 cs is ', cs)
        thk.remove()
        
        mid_out = []
        thk = model.module.cnn[7][i].relu.register_forward_hook(hook_fn)
        _ = model(X)
        # for j in mid_out:
        #     print(j.shape)
        if i == 0: # for some reason, relu layer has 3 output and when in first block, the first relu output is usable
            mid1 = torch.cat([mid_out[j] for j in range(0, 300, 3)])
            cs = (-1 * (spatial.distance.cosine(mid1.flatten().detach().cpu(), img_mid.flatten().detach().cpu()) - 1))
            print(i, 'th seq relu 1st cs is ', cs)
        mid2 = torch.cat([mid_out[j] for j in range(2, 300, 3)])
        cs = (-1 * (spatial.distance.cosine(mid2.flatten().detach().cpu(), img_mid.flatten().detach().cpu()) - 1))
        print(i, 'th seq relu 2nd cs is ', cs)
        thk.remove()
    
    
    

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
group_size = 7 # default 7
# rnn + cnn

data_path = '/notebooks/atk/input/test_videos/apzckowxpy.mp4'
model_path = '/notebooks/atk/models/bi-model_type-baseline_gru_auc_0.150000_ep-10.pth'
image_model_path = '/notebooks/atk/models/image_dectors/all_c40.p' #all_c40 works well, full_raw works fine, the rest are terrible
model_type = 'xception'
logging.basicConfig(filename=time.strftime("%Y%m%d-%H%M%S"), level=logging.INFO)
eps = 4/255 #default 2/255

def rnnbatk():
    for vid_name, (data, y) in video_loader(data_path):
        logging.info('new video %s', vid_name)

        X = data.to(device)
        X = X[0, :5,:,:,:].unsqueeze(0)
        it = 0
        maxiter = 6000
        model = detector.load_model(model_path, device)
        
        GAfunc.cgen_attack(100, X, 1, 60, 0.15, 0.1, maxiter, 20, model, GAfunc.fitness, device, 'rnn')
        return

def rnnatk(l3, sec_phase, max_iters, full_pert):
    for vid_name, (data, y) in video_loader(data_path):
        logging.info('new video %s', vid_name)
        X = data.to(device)
        X = X[0, :100,:,:,:].unsqueeze(0)
        it = 0
        maxiter = max_iters

        modif = torch.Tensor(X.shape).fill_(0.01/255).to(device)
        modifier = torch.nn.Parameter(modif, requires_grad=True)
        min_in = X.min().detach()
        max_in = X.max().detach()
        min_loss = 1e-5
        seq_len = X.shape[1]
        learning_rate = 0.02
        optimizer = torch.optim.Adam([modifier], lr=learning_rate)
        # optimizer = torch.optim.Adagrad([modifier], lr = learning_rate)
        eval_model = detector.load_model(model_path, device)
        img_imp = [1] * (seq_len // group_size + 1)
        img_model = torch.load(image_model_path)
        pivot = 0
        
        # get image detector perturbed input
        # tt = torch.Tensor(X.shape).fill_(0.01/255).to(device) # somehow these effect the modifier
        # ttm = torch.nn.Parameter(tt, requires_grad=True)
        # ttm = image_pert(img_model, X, ttm, 'xception')
        modifier = image_pert(img_model, X, modifier, 'xception')
        ttm = modifier.detach().clone()
        
        f = 0
        for i in range(len(X[0])):
            t, _ = predict_image(img_model, X[0][i], model_type)
            f += t
        logging.info('origin total frame %d, origin total fake frame %d', len(X[0]), f)
        
        global mid_out
        # hook function image
        
        # test_func(img_model, eval_model, X + ttm)
        # print(vars(eval_model.module))
        # exit(0)

        # img_model.model.conv4.register_forward_hook(hook_fn)
        # _ = img_model(X[0])
        # ori_mid = [i for i in mid_out[0]]
        
        # hook function video
        thk = eval_model.module.cnn[7][0].bn3.register_forward_hook(hook_fn)
        # thk = eval_model.module.cnn[4][0].conv1.register_forward_hook(hook_fn)
        _ = eval_model(X + ttm)
        ori_mid = [i for i in mid_out]
        ori_mid = torch.stack(ori_mid).detach()
        thk.remove()
        
        
        while it < maxiter:
            model = detector.load_model(model_path, device)
            model.train()
            # modify for train mode
            for _, m in model.named_modules():
                if 'BatchNorm' in m.__class__.__name__:
                        m = m.eval()
                if 'Dropout' in m.__class__.__name__:
                        m = m.eval()
            
            # # hook for video model
            model.module.cnn[7][0].bn3.register_forward_hook(hook_fn)
            # model.module.cnn[7][2].conv3.register_forward_hook(hook_fn)
            # model.module.cnn[4][0].conv1.register_forward_hook(hook_fn)
            
            
            input_image = autograd.Variable(X, requires_grad=False)
            
            true_image = input_image + modifier
            # true_image = torch.clamp(input_image + torch.clamp(modifier, min=-eps, max=eps), min=0, max=1)
            
            f = 0
            for i in range(len(true_image[0])):
                t, _ = predict_image(img_model, true_image[0][i], model_type)
                f += t
            logging.info('total frame %d, total fake frame %d', len(true_image[0]), f)
            
            mid_out = []
            #Prediction on the adversarial video
            probs, pre_label = model(true_image)
            probs = torch.sigmoid(probs)
            print('iter', it)
            logging.info('iter %d', it)
            print(probs)
            logging.info(probs)

            #extracting the probability of true label 
            zero_array = torch.zeros(2).to(device)
            zero_array[1] = 1
            true_label_onehot = probs*zero_array
            true_label_prob = torch.sum(true_label_onehot, 1)

            #Loss1
            loss1 = -torch.log(1 - true_label_prob + 1e-6) # true_label_prob = sigmoid(prob)
            loss1 = torch.mean(loss1)


            loss2 = torch.sum(torch.sqrt(torch.mean(torch.pow((true_image - input_image), 2), dim=0).mean(dim=2).mean(dim=2).mean(dim=1)))
            # loss2 = torch.sum(torch.sqrt(torch.mean(torch.pow((modifier - ttm), 2), dim=0).mean(dim=2).mean(dim=2).mean(dim=1)))
            norm_frame = torch.mean(torch.abs(modifier), dim=3).mean(dim=3).mean(dim=2) 

            # loss3 for graient's cosine similiarity
            loss3 = 0
            group_level = []

            # for i in range(0, seq_len, group_size):
            #     loss3 += group_cs_center(mid_out[i:min(i + group_size, seq_len)], pivot) * img_imp[i // group_size] # change for the image detector output
            #     group_level.append(sum(mid_out[i:min(i + group_size, seq_len)]))
            # loss3 += group_cs(group_level)
            def abs_tensor(n):
                n = n.unsqueeze(0)
                # print(n.shape)
                n = torch.pow(n, 2)
                # print(n.shape)
                n = torch.mean(n, dim = 0)
                # print(n.shape)
                n = n.mean(dim=1)#1
                # print(n.shape)
                n = n.mean(dim=1)#2
                n = n.mean(dim=0)
                n = torch.sqrt(n)
                n = torch.sum(n)
                return n
            def diversity(p1, p2, f1, f2):
                a1 = f1 + p1
                a2 = f2 + p2
                a1 = a1.unsqueeze(0).unsqueeze(0)
                a2 = a2.unsqueeze(0).unsqueeze(0)
                
                l1_distance = torch.abs(a1 - a2)
    
                # Compute the L2 distance between the perturbed frames
                l2_distance = torch.pow(p1 - p2, 2)
                
                # Compute the temporal dissimilarity between the perturbed frames
                cos_sim = torch.nn.functional.cosine_similarity(p1.flatten().unsqueeze(0), p2.flatten().unsqueeze(0))

                # Combine the three distances using weights
                diversity_score = 0.5 * l1_distance + 0.25 * l2_distance + 0.25 * (1 - cos_sim)
                
                frame_dist = ((f1 - f2) ** 2).sum()

                # Compute the L2 norm of the perturbation
                pert_norm = p1.norm()

                # Return the sum of the frame distance and perturbation norm
                return frame_dist + pert_norm
            diffs = []
            ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            for i in range(len(mid_out) - 1):
                # loss3 += (-1 * (spatial.distance.cosine(mid_out[i].flatten().detach().cpu(), ori_mid[i].flatten().detach().cpu()) - 1))
                # diffs.append((modifier[0][i] - modifier[0][i + 1] - ttm[0][i] + ttm[0][i + 1]))
                # loss3 += abs_tensor(modifier[0][i] - modifier[0][i + 1]) # stronger regularization
                loss3 -= ssim(true_image[0][i].unsqueeze(0), true_image[0][i + 1].unsqueeze(0))
                # loss3 += diversity(modifier[0][i], modifier[0][i + 1], input_image[0][i], input_image[0][i + 1])
            # loss3 = torch.sum(torch.sqrt(torch.mean(torch.pow(torch.stack(diffs).unsqueeze(0), 2), dim=0).mean(dim=2).mean(dim=2).mean(dim=1)))
                
            
            # loss3 /= seq_len # normalize
            # loss3 = -torch.log(torch.tensor(loss3))
            mid_out = torch.stack(mid_out)
            # if torch.all(torch.eq((ori_mid - mid_out), 0)):
            #     mid_out += 1e-12
            # loss3 = torch.sum(torch.sqrt(torch.mean(torch.pow((ori_mid - mid_out), 2), dim=0).mean(dim=2).mean(dim=2).mean(dim=1)))
            print('loss3  ', loss3)
            # aabs = torch.sum(torch.sqrt(torch.mean(torch.pow((modifier - ttm), 2), dim=0).mean(dim=2).mean(dim=2).mean(dim=1)))
            # print('abs', aabs)


            weight_loss2 = 1 #default is 1
            if l3 == True:
                loss = 0.5 * loss1 + weight_loss2 * loss2 + 1 * loss3 # the loss3 cur too high, need to constraint it in first place
                print(loss1, loss2, loss3)
                logging.info('%s, %s, %s', loss1, loss2, loss3)
            else:
                loss = loss1 + weight_loss2 * loss2
                print(loss1, loss2)
                logging.info('%s, %s', loss1, loss2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # check output
            # print('before img pert')
            # print_output(model, img_model, input_image + modifier, group_size, model_type)
            
            

            # do image detector improvement
            # if l3 and sec_phase == True:
            #     for i in range(0, seq_len, group_size):
            #         img_imp[i // group_size] = 1 - 1 / 2 * group_img_dec(true_image[0][i:min(i + 7, seq_len)], img_model, model_type) # this is failed rate, need to lower the implifier
            
            # own method, partial image pert + cs constraint
            
            pivot_step = 0
            if sec_phase == True:
                # for i in range(0, seq_len, group_size):
                #     img_imp[i // group_size] = 1 + 1 * group_img_dec(true_image[0][i:min(i + 7, seq_len)], img_model, model_type)
                if it % 1 == 0:
                    
                    if full_pert:
                        tperts = image_pert(img_model, input_image, torch.tensor(modifier), 'xception')
                        for i in range(len(modifier[0])):
                            with torch.no_grad():
                                modifier[0][i] = tperts[0][i]
                    else:
                        pivot = (pivot + pivot_step) % group_size
                        tperts = image_pert(img_model, input_image, torch.tensor(modifier), 'xception', pivot, group_size)
                        for i in range(len(modifier[0])):
                            if i % group_size == pivot:
                                with torch.no_grad():
                                    modifier[0][i] = tperts[0][i]
                            
            # full image pert, no constraint
            
            # if (l3 and sec_phase == True):
            #     # for i in range(0, seq_len, group_size):
            #     #     img_imp[i // group_size] = 1 + 1 / 2 * group_img_dec(true_image[0][i:min(i + 7, seq_len)], img_model, model_type)
            #     if it % 10 == 0:
            #         modifier = image_pert(img_model, input_image, modifier, 'xception')
            
            # image pert check
            # print('after img pert')
            # print_output(model, img_model, input_image + modifier, group_size, model_type)
            
            
            
            if it % 100 == 0: 
                print (f'Probability for ground truth label : {true_label_prob.detach().cpu().numpy()}')

            break_condition = False
            if loss < min_loss:
                if torch.abs(loss-min_loss) < 0.0001:
                   break_condition = True
                   print ('Aborting early!')
                min_loss = loss

            if it + 1 == maxiter or break_condition:
                print ('Norm frame for each frame: ')
                for pp in range(seq_len):
                    # print the map value for each frame
                    print(str(pp) + ' ' + str((norm_frame[0][pp]).detach().cpu().numpy()))
                    logging.info(str(pp) + ' ' + str((norm_frame[0][pp]).detach().cpu().numpy()))

            # print (f'Prediction for adversarial video: {pre_label.cpu().detach().numpy()}')

            # Empty cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            it += 1
        # check final video output
        true_image = input_image + modifier
        probs, pre_label = eval_model(true_image)
        probs = torch.sigmoid(probs)
        logging.info('final video score %s', probs)
        # check image detector performance
        f = 0
        for i in range(len(true_image[0])):
            t, _ = predict_image(img_model, true_image[0][i], model_type)
            f += t
        logging.info('total frame %d, total fake frame %d', len(true_image[0]), f)
        l21 = torch.sum(torch.sqrt(torch.mean(torch.pow((modifier), 2), dim=0).mean(dim=2).mean(dim=2).mean(dim=1)))
        logging.info('fianl l2,1 nrom %s', l21)
        
        
# for 3dcnn
DETECTOR_WEIGHTS_PATH = 'WIDERFace_DSFD_RES152.fp16.pth'
DETECTOR_THRESHOLD = 0.3
DETECTOR_MIN_SIZE = 512
DETECTOR_MAX_SIZE = 512
DETECTOR_MEAN = (104.0, 117.0, 123.0)
DETECTOR_STD = (1.0, 1.0, 1.0)
DETECTOR_BATCH_SIZE = 16
DETECTOR_STEP = 3

TRACKER_SIGMA_L = 0.3
TRACKER_SIGMA_H = 0.9
TRACKER_SIGMA_IOU = 0.3
TRACKER_T_MIN = 7

VIDEO_MODEL_BBOX_MULT = 1.5
VIDEO_MODEL_MIN_SIZE = 224
VIDEO_MODEL_CROP_HEIGHT = 224
VIDEO_MODEL_CROP_WIDTH = 192
VIDEO_FACE_MODEL_TRACK_STEP = 2
VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH = 7
VIDEO_SEQUENCE_MODEL_TRACK_STEP = 14

VIDEO_SEQUENCE_MODEL_WEIGHTS_PATH = 'efficientnet-b7_ns_seq_aa-original-mstd0.5_100k_v4_cad79a/snapshot_100000.fp16.pth'

VIDEO_BATCH_SIZE = 1
VIDEO_TARGET_FPS = 15
VIDEO_NUM_WORKERS = 0

class TrackSequencesClassifier(object):
    def __init__(self, weights_path):
        model = EfficientNet.from_pretrained('efficientnet-b7', num_classes = 1)

        for module in model.modules():
            if isinstance(module, MBConvBlock):
                if module._block_args.expand_ratio != 1:
                    expand_conv = module._expand_conv
                    seq_expand_conv = detector.SeqExpandConv(expand_conv.in_channels, expand_conv.out_channels,
                                                    VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH)
                    module._expand_conv = seq_expand_conv
        self.model = model.cuda().eval()

        normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.transform = Compose(
            [SmallestMaxSize(VIDEO_MODEL_MIN_SIZE), CenterCrop(VIDEO_MODEL_CROP_HEIGHT, VIDEO_MODEL_CROP_WIDTH),
             normalize, ToTensor()])

        state = torch.load(weights_path, map_location=lambda storage, loc: storage)
        state = {key: value.float() for key, value in state.items()}
        self.model.load_state_dict(state)

    def classify(self, track_sequences):
        track_sequences = [torch.stack([self.transform(image=face)['image'] for face in sequence]) for sequence in
                           track_sequences]
        track_sequences = torch.cat(track_sequences).cuda()
        
        # with torch.no_grad():
        #     # print(self.model(track_sequences))
        #     track_probs = torch.sigmoid(self.model(track_sequences)).flatten().cpu().numpy()
        # return track_probs
    
        eps = 0.006
        
        for param in self.model.parameters():
            param.requires_grad = False
            
        modif = torch.Tensor(track_sequences.shape).fill_(0.01/255).to(device)
        modifier = torch.nn.Parameter(modif, requires_grad=True)
        optimizer = torch.optim.Adam([modifier], lr=0.01)
        
        input_var = autograd.Variable(track_sequences, requires_grad = False).cuda()
        target_var = autograd.Variable(torch.zeros(track_sequences.size(0)), requires_grad = False).cuda() # need to spoof to real, set target to real and minus the gradient later
        it = 0
        maxiter = 40
        pred = np.array([1])
        last = np.array([10 ** 9])
        
        while it < maxiter:
            torch.cuda.empty_cache()
            print('iter', it)
                
            input_var = input_var + modifier
    
            
            loss = 0
            logits = self.model(input_var).flatten()
            loss1 = logits.mean()
            # print(modifier.shape)
            loss2 = torch.sum(torch.sqrt(torch.mean(torch.pow(modifier.unsqueeze(0), 2), dim=0).mean(dim=2).mean(dim=2).mean(dim=1)))
            
            loss3 = group_cs(modifier)# * img_imp[i // group_size]
            
            
            loss = loss1 + loss2 + loss3
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            with torch.no_grad():
                pred = self.model(input_var).flatten()
                score = self.get_score(torch.sigmoid(pred).detach().cpu().numpy())
            if score < 0.02: # early abort
                break
            
            it += 1
        track_probs = torch.sigmoid(pred).detach().cpu().numpy()
        return track_probs, loss2
    def get_score(self, track_probs):
        delta = track_probs - 0.5
        sign = np.sign(delta)
        pos_delta = delta > 0
        neg_delta = delta < 0
        track_probs[pos_delta] = np.clip(0.5 + sign[pos_delta] * np.power(abs(delta[pos_delta]), 0.65), 0.01, 0.99)
        track_probs[neg_delta] = np.clip(0.5 + sign[neg_delta] * np.power(abs(delta[neg_delta]), 0.65), 0.01, 0.99)
        weights = np.power(abs(delta), 1.0) + 1e-4
        video_score = float((track_probs * weights).sum() / weights.sum())
        return video_score

def atk3d(model_path, data_path):


    fd = detector.Detector(os.path.join(model_path, DETECTOR_WEIGHTS_PATH))
    track_sequences_classifier = TrackSequencesClassifier(os.path.join(model_path, VIDEO_SEQUENCE_MODEL_WEIGHTS_PATH))

    dataset = detector.UnlabeledVideoDataset(os.path.join(data_path, 'test_videos'))
    print('Total number of videos: {}'.format(len(dataset)))

    loader = DataLoader(dataset, batch_size=VIDEO_BATCH_SIZE, shuffle=False, num_workers=VIDEO_NUM_WORKERS,
                        collate_fn=lambda X: X,
                        drop_last=False)

    video_name_to_score = {}

    for video_sample in loader:
        frames = video_sample[0]['frames']
        detector_frames = frames[::DETECTOR_STEP]
        video_idx = video_sample[0]['index']
        video_rel_path = dataset.content[video_idx]
        video_name = os.path.basename(video_rel_path)
        print(len(frames))

        if len(frames) == 0:
            video_name_to_score[video_name] = 0.5
            continue

        detections = []
        for start in range(0, len(detector_frames), DETECTOR_BATCH_SIZE):
            end = min(len(detector_frames), start + DETECTOR_BATCH_SIZE)
            detections_batch = fd.detect(detector_frames[start:end])
            for detections_per_frame in detections_batch:
                detections.append({key: value.cpu().numpy() for key, value in detections_per_frame.items()})

        tracks = detector.get_tracks(detections)
        if len(tracks) == 0:
            video_name_to_score[video_name] = 0.5
            continue
        
        pert_sizes = []
        sequence_track_scores = [np.array([])]
        for track in tracks:
            track_sequences = []
            for i, (start_idx, _) in enumerate(
                    track[:-VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH + 1:VIDEO_SEQUENCE_MODEL_TRACK_STEP]):
                assert start_idx >= 0 and start_idx + VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH <= len(frames)
                _, bbox = track[i * VIDEO_SEQUENCE_MODEL_TRACK_STEP + VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH // 2]
                # track_sequences.append(extract_sequence(frames, start_idx, bbox, i % 2 == 0))
                track_sequences = [detector.extract_sequence(frames, start_idx, bbox, i % 2 == 0)]
                one_step = track_sequences_classifier.classify(track_sequences) # return preds and pert_size
                
                pert_sizes.append(one_step[1])
                track_prob = one_step[0]

                sequence_track_scores[0] = np.concatenate([sequence_track_scores[0], track_prob])

        sequence_track_scores = np.concatenate(sequence_track_scores)
        track_probs = sequence_track_scores

        # print(track_probs)
        delta = track_probs - 0.5
        sign = np.sign(delta)
        pos_delta = delta > 0
        neg_delta = delta < 0
        track_probs[pos_delta] = np.clip(0.5 + sign[pos_delta] * np.power(abs(delta[pos_delta]), 0.65), 0.01, 0.99)
        track_probs[neg_delta] = np.clip(0.5 + sign[neg_delta] * np.power(abs(delta[neg_delta]), 0.65), 0.01, 0.99)
        weights = np.power(abs(delta), 1.0) + 1e-4
        video_score = float((track_probs * weights).sum() / weights.sum())
        # print(delta, pos_delta, track_probs)

        video_name_to_score[video_name] = video_score
        print('NUM DETECTION FRAMES: {}, VIDEO SCORE: {}. {}'.format(len(detections), video_name_to_score[video_name],
                                                                 video_rel_path))
        logging.info('NUM DETECTION FRAMES: {}, VIDEO SCORE: {}. {}'.format(len(detections), video_name_to_score[video_name],
                                                                 video_rel_path))
        logging.info(sum(pert_sizes))

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='rnn')
parser.add_argument('--atk', type=str, default='white')
parser.add_argument('--iters',type=int,default=100)
parser.add_argument('--group_size', type = int, default = 7)
parser.add_argument('--l3', action = 'store_true')
parser.add_argument('--sec_phase', action = 'store_true')
parser.add_argument('--full_pert', action = 'store_true')
args = parser.parse_args()

group_size = args.group_size

for arg, value in sorted(vars(args).items()):
    logging.info("Argument %s: %r", arg, value)

if args.atk == 'white':
    if args.model == 'rnn':
        rnnatk(args.l3, args.sec_phase, args.iters, args.full_pert)
    else:
        with open('config.yaml', 'r') as f:
            config = yaml.load(f)
        atk3d(config['MODELS_PATH'], config['DFDC_DATA_PATH'])
else:
    if args.model == 'rnn':
        rnnbatk()
        
        
        
        
#legacy
#indicator = [1] * seq_len
#             #Perturbating the frames
#             true_image = torch.clamp ((modifier[0,0,:,:,:]+input_image[0,0,:,:,:]), min_in, max_in)
#             true_image = torch.unsqueeze(true_image, 0)
            
            
#             for ll in range(seq_len-1):
#                 if indicator[ll+1] != 0:
#                     mask_temp = torch.clamp((indicator[ll+1] * modifier[0,ll+1,:,:,:]+input_image[0,ll+1,:,:,:]), min_in, max_in)
#                 else:
#                     mask_temp = input_image[0,ll+1,:,:,:]
#                 mask_temp = torch.unsqueeze(mask_temp,0)
#                 true_image = torch.cat((true_image, mask_temp),0)
#             true_image = torch.unsqueeze(true_image, 0)

# loss1 custom
#             criterion = AUCLoss(device=device, gamma=0.15, alpha=0.5)
#             loss1 = 0

#             target_var = autograd.Variable(torch.LongTensor([0]).cuda())
#             frame_y = target_var.view(-1, 1)
#             frame_y = frame_y.repeat(1, X.shape[1])
#             frame_y = frame_y.flatten()

#             target_var = autograd.Variable(torch.LongTensor([1]).cuda())
#             target_var = target_var.reshape(-1, 1).float()
#             frame_y = frame_y.reshape(-1, 1).float()

#             video_loss = criterion(probs, target_var)
#             frame_loss = criterion(pre_label, frame_y)
#             loss1 = 0.6 * video_loss + (1 - 0.6) * frame_loss