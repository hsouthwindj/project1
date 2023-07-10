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

data_path = '/kaggle/input/wsvids'
model_path = '/kaggle/input/models/bi-model_type-baseline_gru_auc_0.150000_ep-10.pth'
image_model_path = '/kaggle/input/models/all_c40.p' #all_c40 works well, full_raw works fine, the rest are terrible
model_type = 'xception'
logging.basicConfig(filename=time.strftime("%Y%m%d-%H%M%S"), level=logging.INFO)
eps = 4/255 #default 2/255     

def image_checker(X, model, model_type): # X dim = 5
    f = 0
    if model_type == 'xception':
        for i in range(len(X[0])):
            t, _ = predict_image(model, X[0][i], 'xception')
            f += t
    print('detector %s, origin total frame %d, origin total fake frame %d', model_type, len(X[0]), f)
            
def rnnatk(l3, sec_phase, max_iters, full_pert, reg_type, ws, data_path):
    ct = time.time()
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
        
        model_type = 'xception'
        
        
        image_checker(X, img_model, 'xception')
        # for i in range(len(X[0])):
        #     t, _ = predict_image(mesomodel, X[0][i], 'meso')
        #     print(t)
        # f = 0
        # for i in range(len(X[0])):
        #     t, _ = predict_image(img_model, X[0][i], 'xception')
        #     f += t
        f = 0
        # t = predict_image(cvmodel, X, 'cvit')
        # print(t)
        # logging.info('origin total frame %d, origin total fake frame %d', len(X[0]), f)

        target_pert = dict()
        
        model = detector.load_model(model_path, device)
        model.train()
        # modify for train mode
        for _, m in model.named_modules():
            if 'BatchNorm' in m.__class__.__name__:
                    m = m.eval()
            if 'Dropout' in m.__class__.__name__:
                    m = m.eval()

        print('total frame %d, total fake frame %d', len(modifier[0]), f)
        # print(torch.sum(torch.sqrt(torch.mean(torch.pow(modifier, 2), dim=0).mean(dim=2).mean(dim=2).mean(dim=1))))
        print('video probs', torch.sigmoid(model(X + modifier)[0]))
        logging.info('video probs %s', torch.sigmoid(model(X + modifier)[0]))
        
        
        # TT
#         params = {'kernlen':15, 
#                       'momentum':False,
#                       'move_type':'adj',
#                       'kernel_mode':'gaussian'}
#         attack_method = getattr(attack_methods, 'TemporalTranslation')(model, params=params, steps=10) # all by default, step=10
        
#         true_image = attack_method(X[0], torch.tensor([[1]]))# put the actually label, the attack will do the FGSM
#         true_image = true_image.unsqueeze(0)
        # TT end
    
        # flicking
#         it = 0
#         maxiter = 50
#         while it < maxiter:
#             it += 1
#             true_image = X + modifier
            
#             pred = model(true_image)[0]
            
#             print(pred)
            
#             if pred.mean() < 0:
#                 break
            
#             norm_reg = torch.mean((modifier)**2) + 1e-12

#             perturbation_roll_right = torch.roll(modifier, shifts=1, dims=0)
#             perturbation_roll_left = torch.roll(modifier, shifts=-1, dims=0)

#             # 1st order diff - loss term
#             diff_norm_reg = torch.mean((modifier - perturbation_roll_right)**2) + 1e-12

#             # 2nd order diff - loss term
#             laplacian_norm_reg = torch.mean((-2*modifier + perturbation_roll_right + perturbation_roll_left)**2) + 1e-12
            
#             normal_loss = 0.1*norm_reg + 0.1*diff_norm_reg + 0.1*laplacian_norm_reg
            
#             l_1 = torch.Tensor([0]).cuda()
            
#             to_min_elem = torch.sigmoid(pred)
#             to_max_elem = 1 - to_min_elem
#             loss_margin = 0.05
            
#             l_2 = ((to_min_elem - (to_max_elem - loss_margin))**2) / loss_margin
#             l_3 = to_min_elem - (to_max_elem - loss_margin)
            
#             adv_loss = torch.max(l_1, torch.min(l_2, l_3))
            
#             loss = normal_loss + adv_loss
            
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
        # flickering end
    
        # deepfool => attack by image
        # from utils.deepfool import deepfool
        # advs = []
        # for i in range(len(X[0])):
        #     img = X[0][i]
        #     pert_img = deepfool(img, img_model, num_classes = 2, max_iter = 20)[-1]
        #     advs.append(pert_img)
        # advs = torch.cat(advs).unsqueeze(0)
        
        # deepfool end
        
        # simba
        from utils.simba import SimBA
        advs = []
        simba = SimBA(img_model, 'a', X[0][0].shape[1])
        out = simba.simba_batch(X[0], torch.Tensor([1]).repeat(len(X[0])).long().cuda(), 2500, 64, 8, 4/255, pixel_attack = True)
        advs = out[0].unsqueeze(0)
        
        # simba end
        
        # # hba
        # untargetted_attack_one(model, train_data, attack_idx, x0, y0, init_samples, video_id, args)
        # train_data
        # one_class = Attack_base(model, train_data, attack_idx, x0, y0, output_path, model_name, dataset_name, init_samples)
        # one_class.attack()
        
        
        true_image = advs

        print('time taken', time.time() - ct)
        
        # true_image = X + modifier
        probs, pre_label = eval_model(true_image)
        probs = torch.sigmoid(probs)
        print('final video score %s', probs)
        image_checker(true_image, img_model, 'xception')
        l21 = torch.sum(torch.sqrt(torch.mean(torch.pow((true_image - X), 2), dim=0).mean(dim=2).mean(dim=2).mean(dim=1)))
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
VIDEO_SEQUENCE_MODEL_TRACK_STEP = 7

VIDEO_SEQUENCE_MODEL_WEIGHTS_PATH = 'snapshot_100000.fp16.pth'

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
            [SmallestMaxSize(VIDEO_MODEL_MIN_SIZE), CenterCrop(VIDEO_MODEL_CROP_HEIGHT, VIDEO_MODEL_CROP_WIDTH), ToTensor()])

        state = torch.load(weights_path, map_location=lambda storage, loc: storage)
        state = {key: value.float() for key, value in state.items()}
        self.model.load_state_dict(state)
        
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
        maxiter = 20
        
        image_model_path = '/kaggle/input/models2/all_raw.p' # better for high dimension image
        img_model = torch.load(image_model_path)
        
        print('before atk', img_model(track_sequences)[:10])
        si = 192
        rsf = torchvision.transforms.Resize((si, si))
        
        # print(track_sequences.shape)
         # simba
         # track_sequences[0].shape[1]
        from utils.simba import SimBA
        advs = []
        simba = SimBA(img_model, 'a', si)
        # out = simba.simba_single(rsf(track_sequences)[0], torch.Tensor([1]).long().cuda())
        # print(out)
        out = simba.simba_batch(rsf(track_sequences), torch.Tensor([1]).repeat(len(track_sequences)).long().cuda(), 500, 192, 12, 16/255, pixel_attack = True)
        advs = out[0]
        l2s = out[4]
        
        brsf = torchvision.transforms.Resize((224, 192))
        advs = brsf(advs)
        
        l21 = torch.sum(torch.sqrt(torch.mean(torch.pow((track_sequences - advs).unsqueeze(0), 2), dim=0).mean(dim=2).mean(dim=2).mean(dim=1)))
        #simba end
        
        print('after atk', img_model(advs)[:10])
        fa = [[]]
        for i in range(len(advs)):
            t, _ = predict_image(img_model, advs[i], 'xception')
            if t != 0:
                fa[0].append(start_idx + i)
        pred = self.model(advs).flatten()
        track_probs = torch.sigmoid(pred).detach().cpu().numpy()
        return track_probs, track_sequences - advs, fa
        
    def ori_classify(self, track_sequences):
        track_sequences = [torch.stack([self.transform(image=face)['image'] for face in sequence]) for sequence in
                           track_sequences]
        track_sequences = torch.cat(track_sequences).cuda()
        with torch.no_grad():
            track_probs = torch.sigmoid(self.model(track_sequences)).flatten().cpu().numpy()

        return track_probs
        
def atk3d(model_path, data_path):
    fd = detector.Detector(os.path.join(model_path, DETECTOR_WEIGHTS_PATH))
    track_sequences_classifier = TrackSequencesClassifier(os.path.join(model_path, VIDEO_SEQUENCE_MODEL_WEIGHTS_PATH))

    dataset = detector.UnlabeledVideoDataset(data_path)
    print('Total number of videos: {}'.format(len(dataset)))

    loader = DataLoader(dataset, batch_size=VIDEO_BATCH_SIZE, shuffle=False, num_workers=VIDEO_NUM_WORKERS,
                        collate_fn=lambda X: X,
                        drop_last=False)

    video_name_to_score = {}

    for video_sample in loader:
        frames = video_sample[0]['frames'][:45]
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
        
        # before atk
        for track in tracks:
            track_sequences = []
            for i, (start_idx, _) in enumerate(
                    track[:-VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH + 1:VIDEO_SEQUENCE_MODEL_TRACK_STEP]):
                assert start_idx >= 0 and start_idx + VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH <= len(frames)
                _, bbox = track[i * VIDEO_SEQUENCE_MODEL_TRACK_STEP + VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH // 2]
                # track_sequences.append(extract_sequence(frames, start_idx, bbox, i % 2 == 0))
                track_sequences = [detector.extract_sequence(frames, start_idx, bbox, i % 2 == 0)]
                track_prob = track_sequences_classifier.ori_classify(track_sequences) # return preds and [pert_size, img detecto as fake]
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
        print('after attack')
                
        # for simba, full video mode, nvm, memory not enough
        sequence_track_scores = [np.array([])]
        for track in tracks:
            track_sequences = []
            for i, (start_idx, _) in enumerate(
                    track[:-VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH + 1:7]):
                assert start_idx >= 0 and start_idx + VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH <= len(frames)
                _, bbox = track[i * 7 + VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH // 2]
                # track_sequences.append(extract_sequence(frames, start_idx, bbox, i % 2 == 0))
                track_sequences.append(detector.extract_sequence(frames, start_idx, bbox, i % 2 == 0))
            preds, pert, fake = track_sequences_classifier.classifyn(track_sequences, modifier[start_idx:start_idx + 7], start_idx) # return preds and [pert_size, img detecto as fake]
            f[0] = f[0] | set(fake[0])
            with torch.no_grad():
                modifier = pert
            track_prob = preds
            #print(modifier.shape)
            tsl.append(track_sequences)
        
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
        print('total norm {} and total fake image detected by img detector {} , {} , {}'.format(pert_size, len(f[0]), len(f[1]), len(f[2])))


import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='rnn')
parser.add_argument('--atk', type=str, default='white')
parser.add_argument('--iters',type=int,default=100)
parser.add_argument('--group_size', type = int, default = 7)
parser.add_argument('--l3', action = 'store_true')
parser.add_argument('--sec_phase', action = 'store_true')
parser.add_argument('--full_pert', action = 'store_true')
parser.add_argument('--reg', type=str, default = 'none')
parser.add_argument('--path', type=str)
args = parser.parse_args()

group_size = args.group_size

for arg, value in sorted(vars(args).items()):
    logging.info("Argument %s: %r", arg, value)

if args.atk == 'white':
    if args.model == 'rnn':
        rnnatk(args.l3, args.sec_phase, args.iters, args.full_pert, args.reg, args.group_size, args.path)
    else:
        with open('configforkaggle.yaml', 'r') as f:
            config = yaml.load(f)
        atk3d(config['MODELS_PATH'], config['DFDC_DATA_PATH'])
elif args.atk == 'black':
    if args.model == 'rnn':
        rnnbatk(args.path) 
    else:
        with open('configforkaggle.yaml', 'r') as f:
            config = yaml.load(f)
        batk3d(config['MODELS_PATH'], config['DFDC_DATA_PATH'])    
elif args.atk == 'bench_white':
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
        
        
        
    