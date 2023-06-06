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
from torchmetrics import StructuralSimilarityIndexMeasure, TotalVariation

import sys
sys.path.append('/kaggle/working/project1')

# for VBAD
from attack_related.VBAD.model_wrapper.image_model_wrapper import ResNetFeatureExtractor, DensenetFeatureExtractor, \
    TentativePerturbationGenerator
from attack_related.VBAD.attack.video_attack import untargeted_video_attack
from torchvision import models as tvmodels

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
    loader = detector.rnn_video_loader(device)
    #with open('/notebooks/Deepfake-detection-master/rnnpass') as file:
    #    ok = [line.rstrip()[10:] for line in file]
    if os.path.isdir(url):
        vids = []
        for vid_name in os.listdir(url):
            vid_path = os.path.join(url, vid_name)
            #if vid_path not in ok:
            #    continue
            vids.append((vid_path, loader.load(vid_path)))
            # try:
            #     vids.append((vid_path, loader.load(vid_path)))
            # except:
            #     pass
    else:
        vids = [(url, loader.load(url))]
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
    if not perts.requires_grad:
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
            if fake_score < 0.25 and it > 1:
                break
        else:
            sp_rate = sum([out[i][1] < out[i][0] for i in range(len(out))]) / len(out)
            fake_score = sum([out[i][1] for i in range(len(out))]) / len(out)
            if fake_score < 0.25 and it > 1:
                break
            
        loss = -torch.log(1 - fake_score)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    # print('cur score', fake_score, 'it', it, 'sp_rate', sp_rate)
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

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
group_size = 7 # default 7
# rnn + cnn

data_path = '/notebooks/atk/input/test_videos/apzckowxpy.mp4'
model_path = '/kaggle/input/models/bi-model_type-baseline_gru_auc_0.150000_ep-10.pth'
image_model_path = '/kaggle/input/models/all_c40.p' #all_c40 works well, full_raw works fine, the rest are terrible
model_type = 'xception'
logging.basicConfig(filename=time.strftime("%Y%m%d-%H%M%S"), level=logging.INFO)
eps = 4/255 #default 2/255

def rnn():
    data_path = '/notebooks/atk/input/test_videos'
    model = detector.load_model(model_path, device)
    fake, real = 0, 0
    model.train()
    for vid_name, (data, y) in video_loader(data_path):
        logging.info('new video %s', vid_name)
        X = data.to(device)
        # X = X[0, :100,:,:,:].unsqueeze(0)
        # X = X.unsqueeze(0)
        probs, pre_label = model(X)
        probs = torch.sigmoid(probs)
        logging.info('video %s, fake probability %6f', vid_name, probs)
        if probs > 0.5:
            fake += 1
        else:
            real += 1
    logging.info('total video %d, fake video count %d, fake rate %6f', fake + real, fake, fake / (fake + real))
    

            
            
def rnnbatk(path):
    img_model = torch.load(image_model_path)
    data_path = path
    # vbad partial generator declaration
    def VBAD_items():
        extractors = []
        resnet50 = tvmodels.resnet50(pretrained=True)
        resnet50_extractor = ResNetFeatureExtractor(resnet50, ['fc']).eval().cuda()
        # extractors.append(img_model)
        extractors.append(resnet50_extractor)
        directions_generator = TentativePerturbationGenerator(extractors, part_size=32, preprocess=False,
                                                              device=0)
        return directions_generator
    for vid_name, (data, y) in video_loader(data_path):
        model = detector.load_model(model_path, device)
        model.train()
        # modify for train mode
        for _, m in model.named_modules():
            if 'BatchNorm' in m.__class__.__name__:
                    m = m.eval()
            if 'Dropout' in m.__class__.__name__:
                    m = m.eval()
        X = data.to(device)
        X = X[0, :40,:,:,:].unsqueeze(0)
        X.squeeze_(dim = 0)
        directions_generator = VBAD_items()
        directions_generator.set_untargeted_params(X, random_mask = 1., scale=5.)
        _, _, adv = untargeted_video_attack(model, X, directions_generator,
                                 1, rank_transform=False,
                                 image_split=1,
                                 sub_num_sample=12, sigma=1e-5,
                                 eps=0.05, max_iter=300000,
                                 sample_per_draw=48)
        adv = adv.unsqueeze(0)
        # check final video output
        # print(adv)
        probs, pre_label = model(adv)
        probs = torch.sigmoid(probs)
        logging.info('final video score %s', probs)
        # check image detector performance
        
        f = 0
        for i in range(len(adv[0])):
            t, _ = predict_image(img_model, adv[0][i], model_type)
            f += t
            if t != 0:
                print('f', i, end = ' ')
        logging.info('total frame %d, total fake frame %d', len(adv[0]), f)
        l21 = torch.sum(torch.sqrt(torch.mean(torch.pow((adv - X), 2), dim=0).mean(dim=2).mean(dim=2).mean(dim=1)))
        logging.info('fianl l2,1 nrom %s', l21)

def rnnatk(l3, sec_phase, max_iters, full_pert):
    for vid_name, (data, y) in video_loader(data_path):
        logging.info('new video %s', vid_name)
        X = data.to(device)
        X = X[0, :100,:,:,:].unsqueeze(0)
        it = 0
        maxiter = max_iters

        modif = torch.rand(X.shape).to(device) * (1/255) #torch.Tensor(X.shape).fill_(0.01/255).to(device)
        modifier = torch.nn.Parameter(modif, requires_grad=True)
        min_in = X.min().detach()
        max_in = X.max().detach()
        min_loss = 1e-5
        seq_len = X.shape[1]
        learning_rate = 0.01 # DEFAULT 0.02
        optimizer = torch.optim.Adam([modifier], lr=learning_rate)
        # optimizer = torch.optim.Adagrad([modifier], lr = learning_rate)
        eval_model = detector.load_model(model_path, device)
        img_imp = [1] * (seq_len // group_size + 1)
        img_model = torch.load(image_model_path)
        pivot = 0
        
        # modifier = image_pert(img_model, X, modifier, 'xception')
        # ttm = modifier.detach().clone()
        
        f = 0
        for i in range(len(X[0])):
            t, _ = predict_image(img_model, X[0][i], model_type)
            f += t
        logging.info('origin total frame %d, origin total fake frame %d', len(X[0]), f)
        
        
        while it < maxiter:
            model = detector.load_model(model_path, device)
            model.train()
            # modify for train mode
            for _, m in model.named_modules():
                if 'BatchNorm' in m.__class__.__name__:
                        m = m.eval()
                if 'Dropout' in m.__class__.__name__:
                        m = m.eval()
            
            window_size = 10
            step_size = 8
            # check image detector performance
            f = 0
            ti = X + modifier
            for i in range(len(modifier[0])):
                t, _ = predict_image(img_model, ti[0][i], model_type)
                f += t
                if t != 0:
                    print('f', i, end = ' ')
            print('total frame %d, total fake frame %d', len(modifier[0]), f)
            print(torch.sum(torch.sqrt(torch.mean(torch.pow(modifier, 2), dim=0).mean(dim=2).mean(dim=2).mean(dim=1))))
            loss1 = 0
            loss2 = 0
            loss3 = 0
            reg = 1e-6
            for i in range(0, seq_len - window_size + 1, step_size):
                print('i', i)
                input_image = autograd.Variable(X[0][i:i + window_size], requires_grad=False).unsqueeze(0)
            
                true_image = input_image + modifier[0][i:i + window_size]
            
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
                # loss1 = -torch.log(1 - true_label_prob + 1e-6)
                loss1 += -torch.log(1 - true_label_prob + 1e-6) # true_label_prob = sigmoid(prob)

                
                # loss2 = torch.sum(torch.sqrt(torch.mean(torch.pow((true_image - input_image), 2), dim=0).mean(dim=2).mean(dim=2).mean(dim=1)))
                loss2 += torch.sum(torch.sqrt(torch.mean(torch.pow((true_image - input_image), 2), dim=0).mean(dim=2).mean(dim=2).mean(dim=1)))
                # loss2 = torch.sum(torch.sqrt(torch.mean(torch.pow((modifier - ttm), 2), dim=0).mean(dim=2).mean(dim=2).mean(dim=1)))
                norm_frame = torch.mean(torch.abs(modifier), dim=3).mean(dim=3).mean(dim=2) 

                # loss3 = 0
                print('loss3  ', loss3)
                from lpips import LPIPS
                # target_frame = input_image[0][window_size // 2]
                target_pert = image_pert(img_model, input_image, modifier[0][i:i + window_size].clone().detach().requires_grad_(True), 'xception')
                target_pert = target_pert.unsqueeze(0)
                lpips_distance = 0.0
                lpips_fn = LPIPS(net='alex', verbose=False).to(modifier.device)
                for w in range(window_size):
                    # lpips_distance += lpips_fn(modifier[0][w + i].cuda(), target_pert[0][w].cuda()).item()
                    loss3 += torch.sum(torch.sqrt(torch.mean(torch.pow((modifier[0][w + i] - target_pert[0][w]).unsqueeze(0), 2), dim=0).mean(dim=1).mean(dim=1).mean(dim=0)))
                    # loss3 += img_model(true_image[0][w].unsqueeze(0))[0][1].item()
                # loss3 += (lpips_distance / window_size)
                # loss3 = -torch.log(1 - loss3 + 1e-6)
                
                ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
                tv = TotalVariation().to(device)
                mse = torch.nn.MSELoss()
                treg = 0
                for w in range(window_size - 1):
                    # treg += tv(torch.abs(modifier[0][w + i] - modifier[0][w + i + 1]).unsqueeze(0))
                    # treg += ssim(modifier[0][w + i].unsqueeze(0), modifier[0][w + i + 1].unsqueeze(0))
                    treg += mse(modifier[0][w + i].unsqueeze(0), modifier[0][w + i + 1].unsqueeze(0))
                # reg += -torch.log(treg / window_size)
                reg += (treg / window_size)
            # reg /= 2500

            weight_loss2 = 2 #default is 1
            if l3 == True:
                loss = 0.7 * loss1 + weight_loss2 * loss2 + 1.3 * loss3 + 0.8 * reg
                print(loss1, loss2, loss3, reg)
                logging.info('%s, %s, %s', loss1, loss2, loss3)
            else:
                loss = loss1 + weight_loss2 * loss2
                print(loss1, loss2)
                logging.info('%s, %s', loss1, loss2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
#                 pivot_step = 0
#                 if sec_phase == True:
#                     # for i in range(0, seq_len, group_size):
#                     #     img_imp[i // group_size] = 1 + 1 * group_img_dec(true_image[0][i:min(i + 7, seq_len)], img_model, model_type)
#                     if it % 1 == 0:

#                         if full_pert:
#                             tperts = image_pert(img_model, input_image, torch.tensor(modifier), 'xception')
#                             for i in range(len(modifier[0])):
#                                 with torch.no_grad():
#                                     modifier[0][i] = tperts[0][i]
#                         else:
#                             pivot = (pivot + pivot_step) % group_size
#                             tperts = image_pert(img_model, input_image, torch.tensor(modifier), 'xception', pivot, group_size)
#                             for i in range(len(modifier[0])):
#                                 if i % group_size == pivot:
#                                     with torch.no_grad():
#                                         modifier[0][i] = tperts[0][i]
            
            
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
        true_image = X + modifier
        invTrans = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

        inv_image = invTrans(true_image)
        inv_pert = invTrans(modifier)
        # save image
        cv2.imwrite('img.jpg', inv_image[5].cpu().numpy())
        cv2.imwrite('pert.jpg', inv_pert[5].cpu().numpy())
        
        probs, pre_label = eval_model(true_image)
        probs = torch.sigmoid(probs)
        logging.info('final video score %s', probs)
        # check image detector performance
        f = 0
        for i in range(len(true_image[0])):
            t, _ = predict_image(img_model, true_image[0][i], model_type)
            f += t
            if t != 0:
                print('f', i, end = ' ')
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
            loss3 = 0
            logits = self.model(input_var).flatten()
            loss1 = logits.mean()
            # print(modifier.shape)
            loss2 = torch.sum(torch.sqrt(torch.mean(torch.pow(modifier.unsqueeze(0), 2), dim=0).mean(dim=2).mean(dim=2).mean(dim=1)))
            
            from lpips import LPIPS
            img_model = torch.load(image_model_path)
            target_pert = image_pert(img_model, track_sequences.unsqueeze(0), modifier.clone().detach().requires_grad_(True), 'xception')
            target_pert = target_pert.unsqueeze(0)
            lpips_distance = 0.0
            lpips_fn = LPIPS(net='alex', verbose=False).to(modifier.device)
            window_size = modifier.shape[0]
            for w in range(window_size):
                loss3 += torch.sum(torch.sqrt(torch.mean(torch.pow((modifier[w] - target_pert[0][w]).unsqueeze(0), 2), dim=0).mean(dim=1).mean(dim=1).mean(dim=0)))

            # ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
            # tv = TotalVariation().to(device)
            # mse = torch.nn.MSELoss()
            # treg = 0
            # for w in range(window_size - 1):
            #     # treg += tv(torch.abs(modifier[0][w + i] - modifier[0][w + i + 1]).unsqueeze(0))
            #     # treg += ssim(modifier[0][w + i].unsqueeze(0), modifier[0][w + i + 1].unsqueeze(0))
            #     treg += mse(modifier[0][w].unsqueeze(0), modifier[0][w + 1].unsqueeze(0))
            # # reg += -torch.log(treg / window_size)
            # reg += (treg / window_size)
                 
            
            
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
        # check image detector performance
        f = 0
        for i in range(len(input_var)):
            t, _ = predict_image(img_model, input_var[i], model_type)
            f += t
            if t != 0:
                print('f', i, end = ' ')
        track_probs = torch.sigmoid(pred).detach().cpu().numpy()
        return track_probs, [loss2, f]
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
        fakes = []
        sequence_track_scores = [np.array([])]
        for track in tracks:
            track_sequences = []
            for i, (start_idx, _) in enumerate(
                    track[:-VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH + 1:VIDEO_SEQUENCE_MODEL_TRACK_STEP]):
                assert start_idx >= 0 and start_idx + VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH <= len(frames)
                _, bbox = track[i * VIDEO_SEQUENCE_MODEL_TRACK_STEP + VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH // 2]
                # track_sequences.append(extract_sequence(frames, start_idx, bbox, i % 2 == 0))
                track_sequences = [detector.extract_sequence(frames, start_idx, bbox, i % 2 == 0)]
                preds, [pert_size, fake] = track_sequences_classifier.classify(track_sequences) # return preds and [pert_size, img detecto as fake]
                
                pert_sizes.append(pert_size)
                fakes.append(fake)
                track_prob = preds

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
        logging.info('total norm {} and total fake image detected by img detector {}'.format(sum(pert_sizes), sum(fakes)))

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='rnn')
parser.add_argument('--atk', type=str, default='white')
parser.add_argument('--iters',type=int,default=100)
parser.add_argument('--group_size', type = int, default = 7)
parser.add_argument('--l3', action = 'store_true')
parser.add_argument('--sec_phase', action = 'store_true')
parser.add_argument('--full_pert', action = 'store_true')
parser.add_argument('--path', type=str)
args = parser.parse_args()

group_size = args.group_size

for arg, value in sorted(vars(args).items()):
    logging.info("Argument %s: %r", arg, value)

if args.atk == 'white':
    if args.model == 'rnn':
        rnnatk(args.l3, args.sec_phase, args.iters, args.full_pert)
    else:
        with open('configforkaggle.yaml', 'r') as f:
            config = yaml.load(f)
        atk3d(config['MODELS_PATH'], config['DFDC_DATA_PATH'])
elif args.atk == 'black':
    if args.model == 'rnn':
        rnnbatk(args.path)
else: # only test input with model
    if args.model == 'rnn':
        rnn()
        