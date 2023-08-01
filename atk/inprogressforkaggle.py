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
from attack_related.VBAD.attack.video_attackforkaggle import untargeted_video_attack
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

# for benchmark robust
import torchgeometry as tgm
def compress_decompress(image,factor,cuda=True):
  # :param image: tensor image
  # :param cuda: enables cuda, must be the same parameter as the model
  # using interpolate(input, size, scale_factor, mode, align_corners) to upsample
  # uses either size or scale factor

  image_size = list(image.size())
  image_size = image_size[2:]
  compressed_image = nn.functional.interpolate(image, scale_factor = factor, mode = "bilinear", align_corners = True)
  decompressed_image = nn.functional.interpolate(compressed_image, size = image_size, mode = "bilinear", align_corners = True)

  return decompressed_image



def add_gaussian_noise(image,std,cuda=True):
  # :param image: tensor image
  # :param amount: amount of noise to be added
  # :param cuda: enables cuda, must be the same parameter as the model


  new_image = image + std*torch.randn_like(image)
  new_image = torch.clamp(new_image, min=0, max=1)

  return new_image


def gaussian_blur(image,kernel_size=(11,11),sigma=(10.5, 10.5),cuda=True):
  # smooths the given tensor with a gaussian kernel by convolving it to each channel. 
  # It suports batched operation.
  # :param image: tensor image
  # :param kernel_size (Tuple[int, int]): the size of the kernel
  # :param sigma (Tuple[float, float]): the standard deviation of the kernel

  gauss = tgm.image.GaussianBlur(kernel_size, sigma)

  # blur the image
  img_blur = gauss(image)

  # convert back to numpy. Turned off because I'm returning tensor
  #image_blur = tgm.tensor_to_image(img_blur.byte())

  return img_blur


def translate_image(image, shift_x = 20, shift_y = 20, cuda=True):

  image_size = list(image.size())
  image_size = image_size[2:]

  h, w = image_size[0], image_size[1]  


  #because height indexed from zero
  points_src = torch.FloatTensor([[
    [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1],
  ]])

  # the destination points are the image vertexes
  points_dst = torch.FloatTensor([[
    [0 + shift_x, 0 + shift_y], [w-1+shift_x, 0 + shift_y], [w-1+shift_x, h-1 + shift_y], [0 + shift_x, h-1 + shift_y],]])

  if cuda:
    points_src = points_src.cuda()
    points_dst = points_dst.cuda()
  # compute perspective transform
  print(type(points_src), points_src)
  M = cv2.getPerspectiveTransform(points_src.cpu().detach().numpy(), points_dst.cpu().detach().numpy()) # try to use library in cv
  M = torch.from_numpy(M).float().to(device='cuda')
  # warp the original image by the found transform
  img_warp = tgm.warp_perspective(image, M, dsize=(h, w))

  return img_warp

def _get_transforms(apply_transforms = {"gauss_noise", "gauss_blur", "translation", "resize"}):
        cuda = True
        transform_list = [
            lambda x: x,
        ]

        if "gauss_noise" in apply_transforms:
            transform_list += [
                lambda x: add_gaussian_noise(x, 0.01, cuda = cuda),
            ]

        if "gauss_blur" in apply_transforms:
            transform_list += [
                lambda x: gaussian_blur(x, kernel_size = (5, 5), sigma=(5., 5.), cuda = cuda),
                lambda x: gaussian_blur(x, kernel_size = (5, 5), sigma=(10., 10.), cuda = cuda),
                lambda x: gaussian_blur(x, kernel_size = (7, 7), sigma=(5., 5.), cuda = cuda),
                lambda x: gaussian_blur(x, kernel_size = (7, 7), sigma=(10., 10.), cuda = cuda),
            ]

        if "translation" in apply_transforms:
            transform_list += [
                lambda x: translate_image(x, 10, 10, cuda = cuda),
                lambda x: translate_image(x, 10, -10, cuda = cuda),
                lambda x: translate_image(x, -10, 10, cuda = cuda),
                lambda x: translate_image(x, -10, -10, cuda = cuda),
                lambda x: translate_image(x, 20, 20, cuda = cuda),
                lambda x: translate_image(x, 20, -20, cuda = cuda),
                lambda x: translate_image(x, -20, 10, cuda = cuda),
                lambda x: translate_image(x, -20, -20, cuda = cuda),
            ]

        if "resize" in apply_transforms:
            transform_list += [
                lambda x: compress_decompress(x, 0.1, cuda = cuda),
                lambda x: compress_decompress(x, 0.2, cuda = cuda),
                lambda x: compress_decompress(x, 0.3, cuda = cuda),
            ]

        return transform_list


use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
group_size = 7 # default 7
# rnn + cnn
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
    

            
            
def rnnbatk(data_path, model_path, maxiter):
    img_model = torch.load(image_model_path)
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
        X = X[0, :43,:,:,:].unsqueeze(0)
        X.squeeze_(dim = 0)
        directions_generator = VBAD_items()
        directions_generator.set_untargeted_params(X, random_mask = 1., scale=5.)
        _, _, adv = untargeted_video_attack(model, X, directions_generator,
                                 1, rank_transform=False,
                                 image_split=1,
                                 sub_num_sample=12, sigma=1e-5,
                                 eps=0.05, max_iter=maxiter,
                                 sample_per_draw=48)
        adv = adv.unsqueeze(0)
        # check final video output
        # print(adv)
        probs, pre_label = model(adv)
        probs = torch.sigmoid(probs)
        print('final video score', probs)
        logging.info('final video score %s', probs)
        # check image detector performance
        
        f = 0
        for i in range(len(adv[0])):
            t, _ = predict_image(img_model, adv[0][i], model_type)
            f += t
            if t != 0:
                print('f', i, end = ' ')
        print('total frame and total fake frame', len(adv[0]), f)
        logging.info('total frame %d, total fake frame %d', len(adv[0]), f)
        l21 = torch.sum(torch.sqrt(torch.mean(torch.pow((adv - X), 2), dim=0).mean(dim=2).mean(dim=2).mean(dim=1)))
        print('final l2,1 norm', l21)
        logging.info('fianl l2,1 nrom %s', l21)

def rnnatk(l3, max_iters, ws, data_path, model_path):
    ct = time.time()
    for vid_name, (data, y) in video_loader(data_path):
        print('new video %s', vid_name)
        X = data.to(device)
        X = X[0, :95,:,:,:].unsqueeze(0)
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
        print('origin total frame %d, origin total fake frame %d', len(X[0]), f)
        
        target_pert = dict()
        
        model = detector.load_model(model_path, device)
        model.train()
        # modify for train mode
        for _, m in model.named_modules():
            if 'BatchNorm' in m.__class__.__name__:
                    m = m.eval()
            if 'Dropout' in m.__class__.__name__:
                    m = m.eval()
        while it < maxiter:
            #print('iter', it)
            print('iter %d', it)
            
            
            window_size = ws # default 9
            step_size = window_size - 2 # default 7
            # check image detector performance
            f = 0
            ti = X + modifier
            for i in range(len(modifier[0])):
                t, _ = predict_image(img_model, ti[0][i], model_type)
                f += t
                # if t != 0:
                #     print('f', i, end = ' ')
            #print('total frame %d, total fake frame %d', len(modifier[0]), f)
            #print(torch.sum(torch.sqrt(torch.mean(torch.pow(modifier, 2), dim=0).mean(dim=2).mean(dim=2).mean(dim=1))))
            #print('video probs', torch.sigmoid(model(X + modifier)[0]))
            logging.info('video probs %s', torch.sigmoid(model(X + modifier)[0]))
            loss1 = 0
            loss2 = 0
            loss3 = 0
            for i in range(0, seq_len - window_size + 1, step_size):
                input_image = autograd.Variable(X[0][i:i + window_size], requires_grad=False).unsqueeze(0)
            
                true_image = input_image + modifier[0][i:i + window_size]
            
                #Prediction on the adversarial video
                probs, pre_label = model(true_image)
                probs = torch.sigmoid(probs)
                

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
                # print('loss3  ', loss3)
                from lpips import LPIPS
                # target_frame = input_image[0][window_size // 2]
                if it == 0:
                    target_pert[i] = image_pert(img_model, input_image, modifier[0][i : i + window_size].clone().detach().requires_grad_(True), 'xception')
                    # target_pert[i] = target_pert[i].unsqueeze(0)
                
                lpips_distance = 0.0
                lpips_fn = LPIPS(net='alex', verbose=False).to(modifier.device)
                
                for w in range(window_size):
                    target_prepert = window_size // 4 if w < window_size // 2 else window_size * 3 // 4
                    # lpips_distance += lpips_fn(modifier[0][w + i].cuda(), target_pert[0][w].cuda()).item()
                    loss3 += torch.sum(torch.sqrt(torch.mean(torch.pow((modifier[0][w + i] - target_pert[i][target_prepert]).unsqueeze(0), 2), dim=0).mean(dim=1).mean(dim=1).mean(dim=0)))
                    # loss3 += img_model(true_image[0][w].unsqueeze(0))[0][1].item()
                # loss3 += (lpips_distance / window_size)
                # loss3 = -torch.log(1 - loss3 + 1e-6)
                
            
            # windows size exp var => 2, 1.25, 0.8
            weight_loss2 = 1.25 #default is 1
            if l3 == True:
                loss = 2 * loss1 + weight_loss2 * loss2 + 1.3 * loss3 # default 0.5, 1.25, 1.3
                logging.info('%s, %s, %s', loss1, loss2, loss3)
            else:
                loss = loss1 + weight_loss2 * loss2
                print(loss1, loss2)
                logging.info('%s, %s', loss1, loss2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            if it % 100 == 0: 
                print (f'Probability for ground truth label : {true_label_prob.detach().cpu().numpy()}')

            break_condition = False
            if loss < min_loss:
                if torch.abs(loss-min_loss) < 0.0001:
                   break_condition = True
                   print ('Aborting early!')
                min_loss = loss

            if it + 1 == maxiter or break_condition:
                #print ('Norm frame for each frame: ')
                for pp in range(seq_len):
                    # print the map value for each frame
                    #print(str(pp) + ' ' + str((norm_frame[0][pp]).detach().cpu().numpy()))
                    logging.info(str(pp) + ' ' + str((norm_frame[0][pp]).detach().cpu().numpy()))

            # print (f'Prediction for adversarial video: {pre_label.cpu().detach().numpy()}')

            # Empty cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            it += 1
        #print('time taken', time.time() - ct)
        # check final video output
        true_image = X + modifier
        torchvision.utils.save_image(true_image[0][5], 'img.jpg', normalize = True)
        torchvision.utils.save_image(X[0][5], 'ori.jpg', normalize = True)
        torchvision.utils.save_image(modifier[0][5], 'pert.jpg', normalize = True)
        
        a = X[0][0:60:10]
        b = true_image[0][0:60:10]
        c = modifier[0][0:60:10]*255
        
        torchvision.utils.save_image(torch.cat([a,b]), 'merge.jpg', nrow = 6, normalize = True)
        torchvision.utils.save_image(c, 'perts.jpg', nrow = 6, normalize = True)
        
        probs, pre_label = eval_model(true_image)
        probs = torch.sigmoid(probs)
        logging.info('final video score %s', probs)
        
        true_image = X + modifier
        probs, pre_label = eval_model(true_image)
        probs = torch.sigmoid(probs)
        logging.info('final video score %s', probs)
        print('final video score %s', probs)
        # check image detector performance
        f = 0
        for i in range(len(true_image[0])):
            t, _ = predict_image(img_model, true_image[0][i], model_type)
            f += t
            if t != 0:
                print('f', i, end = ' ')
        logging.info('total frame %d, total fake frame %d', len(true_image[0]), f)
        print('total frame %d, total fake frame %d', len(true_image[0]), f)
        l21 = torch.sum(torch.sqrt(torch.mean(torch.pow((modifier), 2), dim=0).mean(dim=2).mean(dim=2).mean(dim=1)))
        logging.info('fianl l2,1 nrom %s', l21)
        print('fianl l2,1 nrom %s', l21)
        
        
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
VIDEO_SEQUENCE_MODEL_TRACK_STEP = 6 # stride = 7 - 6

VIDEO_SEQUENCE_MODEL_WEIGHTS_PATH = 'snapshot_100000.fp16.pth'

VIDEO_BATCH_SIZE = 1
VIDEO_TARGET_FPS = 15
VIDEO_NUM_WORKERS = 0    
import math  

class TrackSequencesClassifier(object):
    def __init__(self, weights_path, maxiter = 50):
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
            [SmallestMaxSize(VIDEO_MODEL_MIN_SIZE), CenterCrop(VIDEO_MODEL_CROP_HEIGHT, VIDEO_MODEL_CROP_WIDTH), ToTensor()])

        state = torch.load(weights_path, map_location=lambda storage, loc: storage)
        state = {key: value.float() for key, value in state.items()}
        self.model.load_state_dict(state)    
        self.maxiter = maxiter
    def ori_classify(self, track_sequences):
        if isinstance(track_sequences, list):
            track_sequences = [torch.stack([self.transform(image=face)['image'] for face in sequence]) for sequence in
                               track_sequences]
            track_sequences = torch.cat(track_sequences).cuda()
        
        with torch.no_grad():
            track_probs = torch.sigmoid(self.model(track_sequences)).flatten().cpu().numpy()

        return track_probs    
    def classifyn(self, track_sequences, pert, start_idx):
        track_sequences = [torch.stack([self.transform(image=face)['image'] for face in sequence]) for sequence in
                           track_sequences]
        track_sequences = torch.cat(track_sequences).cuda()
        
        modif = torch.Tensor(pert.detach().clone()).fill_(0.01/255).to(device)
        modifier = torch.nn.Parameter(modif, requires_grad=True)
        optimizer = torch.optim.Adam([modifier], lr=0.01)
        ori_sequences = autograd.Variable(track_sequences, requires_grad = False).cuda()
        
        
        for param in self.model.parameters():
            param.requires_grad = False
        
        it = 0
        maxiter = self.maxiter
        
        while it < maxiter:
            f = 7
            torch.cuda.empty_cache()
            print('iter', it)
            perturbed_sequences = ori_sequences + modifier
            
            loss = 0
            loss3 = 0
            logits = self.model(perturbed_sequences).flatten()
            loss1 = torch.sigmoid(logits.mean())
            loss2 = torch.sum(torch.sqrt(torch.mean(torch.pow(modifier.unsqueeze(0), 2), dim=0).mean(dim=2).mean(dim=2).mean(dim=1)))
            
            # from lpips import LPIPS
            if it == 0:
                img_model = torch.load(image_model_path)
                pert_img_model = img_model

                    
            target_pert = image_pert(pert_img_model, track_sequences.unsqueeze(0), modifier.clone().detach().requires_grad_(True), 'xception')
            target_pert = target_pert.unsqueeze(0)
            # lpips_distance = 0.0
            # lpips_fn = LPIPS(net='alex', verbose=False).to(modifier.device)
            window_size = modifier.shape[0]
            for w in range(window_size):
                target_prepert = window_size // 4 if w < window_size // 2 else window_size * 3 // 4
                loss3 += torch.sum(torch.sqrt(torch.mean(torch.pow((modifier[w] - target_pert[0][target_prepert]).unsqueeze(0), 2), dim=0).mean(dim=1).mean(dim=1).mean(dim=0)))
            print(loss1, loss2,loss3)
            loss = 0.1*loss1 + 2*loss2 + 2.5*loss3 # 0.1 4 1
            if math.isnan(loss.item()):
                break
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            
            with torch.no_grad():
                pred = self.model(perturbed_sequences).flatten()
                score = self.get_score(torch.sigmoid(pred).detach().cpu().numpy())
            
            it += 1
            
            # check image detector performance
            if it % 2 == 0:
                fa = [[]]
                for i in range(len(perturbed_sequences)):
                    t, _ = predict_image(img_model, perturbed_sequences[i], 'xception')
                    if t != 0:
                        fa[0].append(start_idx + i)
                for i in fa:
                    if score < 0.2 and len(i) < 3 and loss2 < 0.06:
                        break

        
        with torch.no_grad():
            pred = self.model(track_sequences + modifier).flatten()
            print(pred)
        track_probs = torch.sigmoid(pred).detach().cpu().numpy()
        return track_probs, modifier.detach().clone(), fa
        
    def _nes_gradient_estimator(self, input_var, model, sample_num = 20, sigma = 0.001):
        g = 0
        print('nes')
        for sample in range(sample_num):
            print(sample)
            for transform_fn in _get_transforms({"gauss_noise", "gauss_blur", "resize"}):
                
                rand_noise = torch.randn_like(input_var)
                i1 = input_var + sigma * rand_noise
                i2 = input_var - sigma * rand_noise

                with torch.no_grad():
                    prob1 = torch.mean(model(transform_fn(i1)))
                    prob2 = torch.mean(model(transform_fn(i2)))

                g = g + prob1 * rand_noise
                g = g - prob2 * rand_noise
                g = g.data.detach()
            
        return (1./(2. * sample_num * sigma)) * g
        
    def benchmark_atk(self, track_sequences, attack = 'white'):
        eps = 4/255
        track_sequences = [torch.stack([self.transform(image=face)['image'] for face in sequence]) for sequence in
                           track_sequences]
        track_sequences = torch.cat(track_sequences).cuda()
        
        for param in self.model.parameters():
            param.requires_grad = False
        input_var = autograd.Variable(track_sequences, requires_grad = True).cuda()
        target_var = autograd.Variable(torch.zeros(track_sequences.size(0)), requires_grad = True).cuda() # need to spoof to real, set target to real and minus the gradient later
        it = 0
        pred = np.array([1])
        last = np.array([10 ** 9])
        
        while pred.mean().item() > 0:
            
            torch.cuda.empty_cache()
            if it > 100:
                break
    
            if attack == 'bench_white':
                loss_criterion = nn.CrossEntropyLoss()
                temptotal = []
                for transform_fn in _get_transforms({"gauss_noise", "gauss_blur", "resize"}):
                    loss = 0
                    transformed_img = transform_fn(input_var)
                    logits = self.model(transformed_img).flatten()
                    loss += logits.sum()
                # pred = self.model(input_var).flatten()
                # loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, target_var)
                    loss.backward()
                    temptotal.append(torch.sign(input_var.grad.detach()))
                adv = input_var.detach() - (1/255) * sum(temptotal) /3
                with torch.no_grad():
                    pred = self.model(input_var).flatten()
                #if last.mean() < pred.mean():
                #    pred = last
                #    break
                #last = pred
                # adv = input_var.detach() + (1/255) * torch.sign(input_var.grad.detach())
            elif attack == 'bench_black':
                with torch.no_grad():
                    pred = self.model(input_var).flatten()
                adv = input_var.detach() + (1/255) * self._nes_gradient_estimator(input_var, self.model)
            total_pert = adv - track_sequences
            total_pert = torch.clamp(total_pert, -eps, eps)
            input_adv = track_sequences + total_pert
            input_adv = torch.clamp(input_adv, 0, 1)
            input_var.data = input_adv.detach()
            it += 1
        img_model = torch.load(image_model_path)
        f = 0
        for i in range(len(adv)):
            t, _ = predict_image(img_model, input_var[i], 'xception')
            f += t
        l21 = torch.sum(torch.sqrt(torch.mean(torch.pow((input_var - track_sequences).unsqueeze(0), 2), dim=0).mean(dim=2).mean(dim=2).mean(dim=1)))
        track_probs = torch.sigmoid(pred).detach().cpu().numpy()
        return track_probs, l21, f

    def classify(self, track_sequences):
        track_sequences = [torch.stack([self.transform(image=face)['image'] for face in sequence]) for sequence in
                           track_sequences]
        track_sequences = torch.cat(track_sequences).cuda()
        print(track_sequences.shape)
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
        maxiter = 1
        pred = np.array([1])
        last = np.array([10 ** 9])
        
        while it < self.maxiter:
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
                loss3 += torch.sum(torch.sqrt(torch.mean(torch.pow((modifier[w] - target_pert[0][window_size // 2]).unsqueeze(0), 2), dim=0).mean(dim=1).mean(dim=1).mean(dim=0)))

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

def atk3d(model_path, data_path, maxiter):


    fd = detector.Detector(os.path.join(model_path, DETECTOR_WEIGHTS_PATH))
    track_sequences_classifier = TrackSequencesClassifier(os.path.join(model_path, VIDEO_SEQUENCE_MODEL_WEIGHTS_PATH), maxiter)

    dataset = detector.UnlabeledVideoDataset(os.path.join(data_path))
    print('Total number of videos: {}'.format(len(dataset)))

    loader = DataLoader(dataset, batch_size=VIDEO_BATCH_SIZE, shuffle=False, num_workers=VIDEO_NUM_WORKERS,
                        collate_fn=lambda X: X,
                        drop_last=False)

    video_name_to_score = {}

    for video_sample in loader:
        frames = video_sample[0]['frames'][:100]
        detector_frames = frames[::DETECTOR_STEP]
        video_idx = video_sample[0]['index']
        video_rel_path = dataset.content[video_idx]
        video_name = os.path.basename(video_rel_path)
        print('len', len(frames))

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
        
        sequence_track_scores = [np.array([])]
        f = [set(), set(), set()]
        tsl = []
        modif = torch.rand([len(frames),3,224,192]).to(device) * (1/255) #torch.Tensor(X.shape).fill_(0.01/255).to(device)
        modifier = torch.nn.Parameter(modif)
        # optimizer = torch.optim.Adam([modifier], lr=0.01)
        for track in tracks:
            track_sequences = []
            for i, (start_idx, _) in enumerate(
                    track[:-VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH + 1:6]):
                assert start_idx >= 0 and start_idx + VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH <= len(frames)
                _, bbox = track[i * 6 + VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH // 2]
                # track_sequences.append(extract_sequence(frames, start_idx, bbox, i % 2 == 0))
                track_sequences = [detector.extract_sequence(frames, start_idx, bbox, i % 2 == 0)]
                preds, pert, fake = track_sequences_classifier.classifyn(track_sequences, modifier[start_idx:start_idx + 7], start_idx) # return preds and [pert_size, img detecto as fake]
                f[0] = f[0] | set(fake[0])
                with torch.no_grad():
                    modifier[start_idx:start_idx + 7] = pert
                track_prob = preds
                tsl.append(track_sequences)
                # sequence_track_scores[0] = np.concatenate([sequence_track_scores[0], track_prob])
        
        # verify
        for track in tracks:
            track_sequences = []
            for i, (start_idx, _) in enumerate(
                    track[:-VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH + 1:VIDEO_SEQUENCE_MODEL_TRACK_STEP]):
                assert start_idx >= 0 and start_idx + VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH <= len(frames)
                _, bbox = track[i * VIDEO_SEQUENCE_MODEL_TRACK_STEP + VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH // 2]
                # track_sequences.append(extract_sequence(frames, start_idx, bbox, i % 2 == 0))
                track_sequences = [detector.extract_sequence(frames, start_idx, bbox, i % 2 == 0)]
                preds = track_sequences_classifier.ori_classify(track_sequences) # return preds and [pert_size, img detecto as fake]
                

                sequence_track_scores[0] = np.concatenate([sequence_track_scores[0], track_prob])
        
        pert_size = torch.sum(torch.sqrt(torch.mean(torch.pow(modifier.unsqueeze(0), 2), dim=0).mean(dim=2).mean(dim=2).mean(dim=1)))
        # print(len(modifier))
        # print(modifier)
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
        logging.info('total norm {} and total fake image detected by img detector {}'.format(pert_size, len(f[0])))       

def batk3d(model_path, data_path, maxiter):
    fd = detector.Detector(os.path.join(model_path, DETECTOR_WEIGHTS_PATH))
    track_sequences_classifier = TrackSequencesClassifier(os.path.join(model_path, VIDEO_SEQUENCE_MODEL_WEIGHTS_PATH))

    dataset = detector.UnlabeledVideoDataset(data_path)
    print('Total number of videos: {}'.format(len(dataset)))

    loader = DataLoader(dataset, batch_size=VIDEO_BATCH_SIZE, shuffle=False, num_workers=VIDEO_NUM_WORKERS,
                        collate_fn=lambda X: X,
                        drop_last=False)

    video_name_to_score = {}

    for video_sample in loader:
        frames = video_sample[0]['frames'][:100]
        detector_frames = frames[::DETECTOR_STEP]
        video_idx = video_sample[0]['index']
        video_rel_path = dataset.content[video_idx]
        video_name = os.path.basename(video_rel_path)
        print('len', len(frames))

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
        
        sequence_track_scores = [np.array([])]
        # optimizer = torch.optim.Adam([modifier], lr=0.01)
        track_sequences = []
        for track in tracks:
            track_sequences = []
            for i, (start_idx, _) in enumerate(
                    track[:-VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH + 1:7]):
                assert start_idx >= 0 and start_idx + VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH <= len(frames)
                _, bbox = track[i * 7 + VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH // 2]
                # track_sequences.append(extract_sequence(frames, start_idx, bbox, i % 2 == 0))
                track_sequences.append(detector.extract_sequence(frames, start_idx, bbox, i % 2 == 0))
                
                
        track_sequences = [torch.stack([track_sequences_classifier.transform(image=face)['image'] for face in sequence]) for sequence in
                           track_sequences]
        X = torch.cat(track_sequences).cuda().unsqueeze(0)
        image_model_path = '/kaggle/input/models2/all_raw.p'
        # do atk
        img_model = torch.load(image_model_path)
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
        X = X[0, :7,:,:,:].unsqueeze(0)
        X.squeeze_(dim = 0)
        directions_generator = VBAD_items()
        directions_generator.set_untargeted_params(X, random_mask = 1., scale=5.)
        advs = []
        for i in range(len(X) // 7):
            _, _, adv = untargeted_video_attack(track_sequences_classifier, X[7*i:7*i + 7], directions_generator,
                                     1, rank_transform=False,
                                     image_split=1,
                                     sub_num_sample=12, sigma=1e-5,
                                     eps=0.05, max_iter=maxiter,
                                     sample_per_draw=48, vc = 'c3d')
            adv = adv.unsqueeze(0)
            advs.append(adv)
        adv = torch.cat(advs, dim = 1)
        # check final video output
        # print(adv)
        probs = track_sequences_classifier.get_score(track_sequences_classifier.ori_classify(adv.squeeze(0)))
        logging.info('final video score %s', probs)
        print('final video score', probs)
        # check image detector performance

        f = 0
        for i in range(len(adv[0])):
            t, _ = predict_image(img_model, adv[0][i], model_type)
            f += t
            if t != 0:
                print('f', i, end = ' ')
        print('=======================================')
        logging.info('total frame %d, total fake frame %d', len(adv[0]), f)
        print('total frames and total fake frames', len(adv[0]), f)
        l21 = torch.sum(torch.sqrt(torch.mean(torch.pow((adv - X), 2), dim=0).mean(dim=2).mean(dim=2).mean(dim=1)))
        logging.info('fianl l2,1 nrom %s', l21)
        print('norm ', l21)

def benchatk(model_path, data_path):
    fd = detector.Detector(os.path.join(model_path, DETECTOR_WEIGHTS_PATH))
    track_sequences_classifier = TrackSequencesClassifier(os.path.join(model_path, VIDEO_SEQUENCE_MODEL_WEIGHTS_PATH))

    dataset = detector.UnlabeledVideoDataset(data_path)
    print('Total number of videos: {}'.format(len(dataset)))

    loader = DataLoader(dataset, batch_size=VIDEO_BATCH_SIZE, shuffle=False, num_workers=VIDEO_NUM_WORKERS,
                        collate_fn=lambda X: X,
                        drop_last=False)

    video_name_to_score = {}

    for video_sample in loader:
        frames = video_sample[0]['frames'][:100]
        detector_frames = frames[::DETECTOR_STEP]
        video_idx = video_sample[0]['index']
        video_rel_path = dataset.content[video_idx]
        video_name = os.path.basename(video_rel_path)
        print('len', len(frames))

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
        
        sequence_track_scores = [np.array([])]
        f = 0
        l21 = 0
        for track in tracks:
            track_sequences = []
            for i, (start_idx, _) in enumerate(
                    track[:-VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH + 1:7]):
                assert start_idx >= 0 and start_idx + VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH <= len(frames)
                _, bbox = track[i * 7 + VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH // 2]
                track_sequences = [detector.extract_sequence(frames, start_idx, bbox, i % 2 == 0)]
                one_step, pert_size, sf = track_sequences_classifier.benchmark_atk(track_sequences, attack = args.atk)
                f += sf
                l21 += pert_size
                sequence_track_scores[0] = np.concatenate([sequence_track_scores[0], one_step])

        sequence_track_scores = np.concatenate(sequence_track_scores)
        track_probs = sequence_track_scores

        delta = track_probs - 0.5
        sign = np.sign(delta)
        pos_delta = delta > 0
        neg_delta = delta < 0
        track_probs[pos_delta] = np.clip(0.5 + sign[pos_delta] * np.power(abs(delta[pos_delta]), 0.65), 0.01, 0.99)
        track_probs[neg_delta] = np.clip(0.5 + sign[neg_delta] * np.power(abs(delta[neg_delta]), 0.65), 0.01, 0.99)
        weights = np.power(abs(delta), 1.0) + 1e-4
        video_score = float((track_probs * weights).sum() / weights.sum())

        video_name_to_score[video_name] = video_score
        print('NUM DETECTION FRAMES: {}, VIDEO SCORE: {}. {}'.format(len(detections), video_name_to_score[video_name],
                                                                     video_rel_path))
        print('fianl l21 norm {}'.format(l21))
        print('total fake frame ', f)




#======================== main function ===========================

import argparse

# be caution to the path
# the attack for rnn+cnn, the path for data(dpath) and be a file or directory, the model path(mpath) needs to be a fixed url to the model
# 
# the attack for 3dcnn, the path for both data and model needs to be a directory since it has its own data loader and such, i don't want to mess with it

# you can run this on kaggle if the pathes are set correctly
# default rnn+cnn model path
model_path = '/kaggle/input/models/bi-model_type-baseline_gru_auc_0.150000_ep-10.pth'
# default 3dcnn model path
model_path_3d = '/kaggle/input/models'
# default data folder
data_path = '/kaggle/input/small-vids1-8'

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='rnn') # target video model type, rnn+cnn or 3dcnn
parser.add_argument('--atk', type=str, default='white') # white or black box attack
parser.add_argument('--iters',type=int,default=100) # maximum iterations, only works for white box since black box setting are set to default VBAD setting
parser.add_argument('--group_size', type = int, default = 7) # window size (white box only)
parser.add_argument('--l3', action = 'store_true')  # whether to use loss3, you probaably always turn this on so you can ignore this
parser.add_argument('--mpath', type=str)
parser.add_argument('--dpath', type=str)
args = parser.parse_args()

# also, for 3dcnn, the window size is actually fixed due to the model structure
# but still you could modify the window stride as well

group_size = args.group_size

for arg, value in sorted(vars(args).items()):
    logging.info("Argument %s: %r", arg, value)
    


if args.atk == 'white':
    if args.model == 'rnn':
        rnnatk(args.l3, args.iters, args.group_size, data_path, model_path)
    else:
        atk3d(model_path_3d, data_path, args.iters)
elif args.atk == 'black':
    if args.model == 'rnn':
        rnnbatk(data_path, model_path, args.iters) 
    else:
        batk3d(model_path_3d, data_path, args.iters)   
elif args.atk == 'bench_white':                                    # below parts are for testing other attack, ignore these
    if args.model == 'c3d':
        with open('configforkaggle.yaml', 'r') as f:
            config = yaml.load(f)
        benchatk(config['MODELS_PATH'], config['DFDC_DATA_PATH'])
elif args.atk == 'bench_black':
    if args.model == 'c3d':
        with open('configforkaggle.yaml', 'r') as f:
            config = yaml.load(f)
        benchatk(config['MODELS_PATH'], config['DFDC_DATA_PATH'])
else:
    if args.model == 'rnn':
        rnnbatk()
        
        
        
    