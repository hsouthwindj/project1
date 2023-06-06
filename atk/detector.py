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
import cv2
from facenet_pytorch.models.mtcnn import MTCNN
from sklearn.metrics import accuracy_score

import sys
sys.path.append('/notebooks/atk/')
from model.model import Baseline
from utils.aucloss import AUCLoss
import time


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

RNNCNN_MODEL_PATH = 'bi-model_type-baseline_gru_auc_0.150000_ep-10.pth'

class UnlabeledVideoDataset(Dataset):
    def __init__(self, root_dir, content=None):
        self.root_dir = os.path.normpath(root_dir)
        if content is not None:
            self.content = content
        else:
            self.content = []
            for path in glob.iglob(os.path.join(self.root_dir, '**', '*.mp4'), recursive=True):
                rel_path = path[len(self.root_dir) + 1:]
                self.content.append(rel_path)
            self.content = sorted(self.content)

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        rel_path = self.content[idx]
        path = os.path.join(self.root_dir, rel_path)

        sample = {
            'frames': [],
            'index': idx
        }

        capture = cv2.VideoCapture(path)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count == 0:
            return sample

        fps = int(capture.get(cv2.CAP_PROP_FPS))
        video_step = round(fps / VIDEO_TARGET_FPS)
        if video_step == 0:
            return sample

        for i in range(frame_count):
            capture.grab()
            if i % video_step != 0:
                continue
            ret, frame = capture.retrieve()
            if not ret:
                continue

            sample['frames'].append(frame)

        return sample


class Detector(object):
    def __init__(self, weights_path):
        self.model = SSD('test')
        self.model.cuda().eval()

        state = torch.load(weights_path, map_location=lambda storage, loc: storage)
        state = {key: value.float() for key, value in state.items()}
        self.model.load_state_dict(state)

        self.transform = GeneralizedRCNNTransform(DETECTOR_MIN_SIZE, DETECTOR_MAX_SIZE, DETECTOR_MEAN, DETECTOR_STD)
        self.transform.eval()

    def detect(self, images):
        images = torch.stack([torch.from_numpy(image).cuda() for image in images])
        images = images.transpose(1, 3).transpose(2, 3).float()
        original_image_sizes = [img.shape[-2:] for img in images]
        images, _ = self.transform(images, None)
        with torch.no_grad():
            detections_batch = self.model(images.tensors).cpu().numpy()
        result = []
        for detections, image_size in zip(detections_batch, images.image_sizes):
            scores = detections[1, :, 0]
            keep_idxs = scores > DETECTOR_THRESHOLD
            detections = detections[1, keep_idxs, :]
            detections = detections[:, [1, 2, 3, 4, 0]]
            detections[:, 0] *= image_size[1]
            detections[:, 1] *= image_size[0]
            detections[:, 2] *= image_size[1]
            detections[:, 3] *= image_size[0]
            result.append({
                'scores': torch.from_numpy(detections[:, 4]),
                'boxes': torch.from_numpy(detections[:, :4])
            })

        result = self.transform.postprocess(result, images.image_sizes, original_image_sizes)
        return result


def get_tracks(detections):
    if len(detections) == 0:
        return []

    converted_detections = []
    frame_bbox_to_face_idx = {}
    for i, detections_per_frame in enumerate(detections):
        converted_detections_per_frame = []
        for j, (bbox, score) in enumerate(zip(detections_per_frame['boxes'], detections_per_frame['scores'])):
            bbox = tuple(bbox.tolist())
            frame_bbox_to_face_idx[(i, bbox)] = j
            converted_detections_per_frame.append({'bbox': bbox, 'score': score})
        converted_detections.append(converted_detections_per_frame)

    tracks = track_iou(converted_detections, TRACKER_SIGMA_L, TRACKER_SIGMA_H, TRACKER_SIGMA_IOU, TRACKER_T_MIN)
    tracks_converted = []
    for track in tracks:
        start_frame = track['start_frame'] - 1
        bboxes = np.array(track['bboxes'], dtype=np.float32)
        frame_indices = np.arange(start_frame, start_frame + len(bboxes)) * DETECTOR_STEP
        interp_frame_indices = np.arange(frame_indices[0], frame_indices[-1] + 1)
        interp_bboxes = np.zeros((len(interp_frame_indices), 4), dtype=np.float32)
        for i in range(4):
            interp_bboxes[:, i] = np.interp(interp_frame_indices, frame_indices, bboxes[:, i])

        track_converted = []
        for frame_idx, bbox in zip(interp_frame_indices, interp_bboxes):
            track_converted.append((frame_idx, bbox))
        tracks_converted.append(track_converted)

    return tracks_converted


class SeqExpandConv(nn.Module):
    def __init__(self, in_channels, out_channels, seq_length):
        super(SeqExpandConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False)
        self.seq_length = seq_length

    def forward(self, x):
        batch_size, in_channels, height, width = x.shape
        x = x.view(batch_size // self.seq_length, self.seq_length, in_channels, height, width)
        x = self.conv(x.transpose(1, 2).contiguous()).transpose(2, 1).contiguous()
        x = x.flatten(0, 1)
        return x


class TrackSequencesClassifier(object):
    def __init__(self, weights_path):
        model = EfficientNet.from_pretrained('efficientnet-b7', num_classes = 1)

        for module in model.modules():
            if isinstance(module, MBConvBlock):
                if module._block_args.expand_ratio != 1:
                    expand_conv = module._expand_conv
                    seq_expand_conv = SeqExpandConv(expand_conv.in_channels, expand_conv.out_channels,
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
        with torch.no_grad():
            # print(self.model(track_sequences))
            track_probs = torch.sigmoid(self.model(track_sequences)).flatten().cpu().numpy()
        return track_probs


def extract_sequence(frames, start_idx, bbox, flip):
    frame_height, frame_width, _ = frames[start_idx].shape
    xmin, ymin, xmax, ymax = bbox
    width = xmax - xmin
    height = ymax - ymin
    xcenter = xmin + width / 2
    ycenter = ymin + height / 2
    width = width * VIDEO_MODEL_BBOX_MULT
    height = height * VIDEO_MODEL_BBOX_MULT
    xmin = xcenter - width / 2
    ymin = ycenter - height / 2
    xmax = xmin + width
    ymax = ymin + height

    xmin = max(int(xmin), 0)
    xmax = min(int(xmax), frame_width)
    ymin = max(int(ymin), 0)
    ymax = min(int(ymax), frame_height)

    sequence = []
    for i in range(VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH):
        face = cv2.cvtColor(frames[start_idx + i][ymin:ymax, xmin:xmax], cv2.COLOR_BGR2RGB)
        sequence.append(face)

    if flip:
        sequence = [face[:, ::-1] for face in sequence]

    return sequence

def _3dcnn(model_path, data_path):
    

    detector = Detector(os.path.join(model_path, DETECTOR_WEIGHTS_PATH))
    track_sequences_classifier = TrackSequencesClassifier(os.path.join(model_path, VIDEO_SEQUENCE_MODEL_WEIGHTS_PATH))

    dataset = UnlabeledVideoDataset(os.path.join(data_path, 'test_videos'))
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
            detections_batch = detector.detect(detector_frames[start:end])
            for detections_per_frame in detections_batch:
                detections.append({key: value.cpu().numpy() for key, value in detections_per_frame.items()})

        tracks = get_tracks(detections)
        if len(tracks) == 0:
            video_name_to_score[video_name] = 0.5
            continue

        sequence_track_scores = [np.array([])]
        for track in tracks:
            track_sequences = []
            for i, (start_idx, _) in enumerate(
                    track[:-VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH + 1:VIDEO_SEQUENCE_MODEL_TRACK_STEP]):
                assert start_idx >= 0 and start_idx + VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH <= len(frames)
                _, bbox = track[i * VIDEO_SEQUENCE_MODEL_TRACK_STEP + VIDEO_SEQUENCE_MODEL_SEQUENCE_LENGTH // 2]
                track_sequences.append(extract_sequence(frames, start_idx, bbox, i % 2 == 0))
                # track_sequences = [extract_sequence(frames, start_idx, bbox, i % 2 == 0)]
                # sequence_track_scores[0] = np.concatenate([sequence_track_scores[0], track_sequences_classifier.classify(track_sequences)])
            sequence_track_scores.append(track_sequences_classifier.classify(track_sequences))

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

        
        
def load_model(restore_from, device):
    model = Baseline(use_gru=True, bi_branch=True)

    model.to(device)

    device_count = torch.cuda.device_count()
    # if device_count > 1:
    #     print('Using {} GPUs'.format(device_count))
    model = nn.DataParallel(model)

    if restore_from is not None:
        ckpt = torch.load(restore_from, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        print('Model is loaded from %s' % restore_from)

    model.eval()

    return model

def _bbox_in_img(img, bbox):
    """
    check whether the bbox is inner an image.
    :param img: (3-d np.ndarray), image
    :param bbox: (list) [x, y, width, height]
    :return: (bool), whether bbox in image size.
    """
    if not isinstance(img, np.ndarray):
        raise ValueError("input image should be ndarray!")
    if len(img.shape) != 3:
        raise ValueError("input image should be (w,h,c)!")
    h = img.shape[0]
    w = img.shape[1]
    x_in = 0 <= bbox[0] <= w
    y_in = 0 <= bbox[1] <= h
    x1_in = 0 <= bbox[0] + bbox[2] <= w
    y1_in = 0 <= bbox[1] + bbox[3] <= h
    return x_in and y_in and x1_in and y1_in


def _enlarged_bbox(bbox, expand):
    """
    enlarge a bbox by given expand param.
    :param bbox: [x, y, width, height]
    :param expand: (tuple) (h,w), expanded pixels in height and width. if (int), same value in both side.
    :return: enlarged bbox
    """
    if isinstance(expand, int):
        expand = (expand, expand)
    s_0, s_1 = bbox[1], bbox[0]
    e_0, e_1 = bbox[1] + bbox[3], bbox[0] + bbox[2]
    x = s_1 - expand[1]
    y = s_0 - expand[0]
    x1 = e_1 + expand[1]
    y1 = e_0 + expand[0]
    width = x1 - x
    height = y1 - y
    return x, y, width, height


def _box_mode_cvt(bbox):
    """
    convert box from FCOS([xyxy], float) output to [x, y, width, height](int).
    :param bbox: (dict), an output from FCOS([x, y, x1, y1], float).
    :return: (list[int]), a box with [x, y, width, height] format.
    """
    if bbox is None:
        raise ValueError("There is no box in the dict!")
    # FCOS box format is [x, y, x1, y1]
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    cvt_box = [int(bbox[0]), int(bbox[1]), max(int(w), 0), max(int(h), 0)]
    return cvt_box


def crop_bbox(img, bbox):
    """
    crop an image by giving exact bbox.
    :param img:
    :param bbox: [x, y, width, height]
    :return: cropped image
    """
    if not _bbox_in_img(img, bbox):
        raise ValueError("bbox is out of image size!img size: {0}, bbox size: {1}".format(img.shape, bbox))
    s_0 = bbox[1]
    s_1 = bbox[0]
    e_0 = bbox[1] + bbox[3]
    e_1 = bbox[0] + bbox[2]
    cropped_img = img[s_0:e_0, s_1:e_1, :]
    return cropped_img

def face_boxes_post_process(img, box, expand_ratio):
    """
    enlarge and crop the face patch from image
    :param img: ndarray, 1 frame from video
    :param box: output of MTCNN
    :param expand_ratio: default: 1.3
    :return:
    """
    box = [max(b, 0) for b in box]
    box_xywh = _box_mode_cvt(box)
    expand_w = int((box_xywh[2] * (expand_ratio - 1)) / 2)
    expand_h = int((box_xywh[3] * (expand_ratio - 1)) / 2)
    enlarged_box = _enlarged_bbox(box_xywh, (expand_h, expand_w))
    try:
        res = crop_bbox(img, enlarged_box)
    except ValueError:
        try:
            res = crop_bbox(img, box_xywh)
        except ValueError:
            return img
    return res

def detect_face(frame, face_detector):
        boxes, _ = face_detector.detect(frame)
        if boxes is not None:
            best_box = boxes[0, :]
            best_face = face_boxes_post_process(frame, best_box, expand_ratio=1.33)
            return best_face
        else:
            return None

class rnn_video_loader():
    def __init__(self, device):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.face_detector = MTCNN(margin=0, keep_all=False, select_largest=False, thresholds=[0.6, 0.7, 0.7],
                              min_face_size=35, factor=0.8, device=device).eval()
    def load(self, path):
        print('loading filename', path)
        video_fd = cv2.VideoCapture(path)
        if not video_fd.isOpened():
            print('problem of reading video')
            return

        frame_index = 0
        faces = []
        success, frame = video_fd.read()
        while success:
            cropped_face = detect_face(frame, self.face_detector)
            if cropped_face is None or len(cropped_face) == 0:
                success, frame = video_fd.read()
                continue
            cropped_face = cv2.resize(cropped_face, (64, 64))
            if cropped_face is not None:
                cropped_face = self.transform(cropped_face)
                faces.append(cropped_face)
            frame_index += 1
            success, frame = video_fd.read()
        video_fd.release()
        print('video frame length:', frame_index)
        faces = torch.stack(faces, dim=0)
        faces = torch.unsqueeze(faces, 0)
        y = torch.ones(frame_index).type(torch.IntTensor)
        return faces, y

def load_data(path, device):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    face_detector = MTCNN(margin=0, keep_all=False, select_largest=False, thresholds=[0.6, 0.7, 0.7],
                          min_face_size=35, factor=0.8, device=device).eval()
    video_fd = cv2.VideoCapture(path)
    if not video_fd.isOpened():
        print('problem of reading video')
        return

    frame_index = 0
    faces = []
    success, frame = video_fd.read()
    
    while success:
        cropped_face = detect_face(frame, face_detector)
        cropped_face = cv2.resize(cropped_face, (64, 64))
        if cropped_face is not None:
            cropped_face = transform(cropped_face)
            faces.append(cropped_face)
        frame_index += 1
        success, frame = video_fd.read()
    
    video_fd.release()
    print('video frame length:', frame_index)
    faces = torch.stack(faces, dim=0)
    faces = torch.unsqueeze(faces, 0)
    y = torch.ones(frame_index).type(torch.IntTensor)
    return faces, y

def _rnncnn(model_path, data_path):
    frame_y_gd = []
    y_pred = []
    frame_y_pred = []
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    model = load_model(os.path.join(model_path, RNNCNN_MODEL_PATH), device)
    data, y = load_data('/notebooks/atk/input/test_videos/aaqaifqrwn.mp4', device)
    X = data.to(device)
    y_, cnn_y = model(X)
    y_ = torch.sigmoid(y_)
    print(y_)
    frame_y_ = torch.sigmoid(cnn_y)
    frame_y_gd += y.detach().numpy().tolist()
    frame_y_pred += frame_y_.detach().cpu().numpy().tolist()
    frame_y_pred = torch.tensor(frame_y_pred)
    frame_y_pred = [0 if i < 0.5 else 1 for i in frame_y_pred]
    test_frame_acc = accuracy_score(frame_y_gd, frame_y_pred)
    print('video is fake:', (y_ >= 0.5).item())
    print('frame level acc:', test_frame_acc)
        
def main():
    with open('config.yaml', 'r') as f:
        config = yaml.load(f)
    if config['attack'] == '3dcnn':
        _3dcnn(config['MODELS_PATH'], config['DFDC_DATA_PATH'])
    elif config['attack'] == 'rnn+cnn':
        _rnncnn(config['MODELS_PATH'], config['DFDC_DATA_PATH'])
    else:
        print('wrong attack type')

# main()