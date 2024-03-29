import torch
from torch import nn

from torchvision import models as Models

from dct.imagenet.gate import GateModule192
from dct.utils import kaiming_init, constant_init
from models.convlstm import ConvLSTM
from models.convGRU import ConvGRU
from models import resnet

from dct.imagenet.resnet import *


class Baseline(nn.Module):

    def __init__(self, use_gru=False, bi_branch=False, rnn_hidden_layers=3, rnn_hidden_nodes=256,
                 num_classes=1, bidirectional=False, dct=False, inputgate=False):

        super(Baseline, self).__init__()

        self.rnn_hidden_layers = rnn_hidden_layers
        self.rnn_hidden_nodes = rnn_hidden_nodes
        self.num_classes = num_classes
        self.bi_branch = bi_branch
        self.inputgate = inputgate

        if not dct:
            pretrained_cnn = Models.resnet50(pretrained=True)
            cnn_layers = list(pretrained_cnn.children())[:-1]
        else:
            pretrained_cnn = ResNetDCT_Upscaled_Static(channels=192, pretrained=True)
            cnn_layers = list(pretrained_cnn.children())[:-2]

        self.cnn = nn.Sequential(*cnn_layers)
        rnn_params = {
            'input_size': pretrained_cnn.fc.in_features,
            'hidden_size': self.rnn_hidden_nodes,
            'num_layers': self.rnn_hidden_layers,
            'batch_first': True,
            'bidirectional': bidirectional
        }

        if bidirectional:
            fc_in = 2 * rnn_hidden_nodes
        else:
            fc_in = rnn_hidden_nodes

        self.rnn = (nn.GRU if use_gru else nn.LSTM)(**rnn_params)

        self.fc_cnn = nn.Linear(fc_in, num_classes)

        self.global_pool = nn.AdaptiveAvgPool2d(16)

        self.fc_rnn = nn.Linear(256, self.num_classes)

        if inputgate:
            self.inp_GM = GateModule192()
            self._initialize_weights()

    def forward(self, x_3d):

        cnn_embedding_out = []
        cnn_pred = []
        frame_num = x_3d.size(1)
        gates = []

        for t in range(frame_num):
            if self.inputgate:
                x, gate_activations = self.inp_GM(x_3d[:, t, :, :, :])
                gates.append(gate_activations)
            x = self.cnn(x_3d[:, t, :, :, :])
            x = torch.flatten(x, start_dim=1)
            cnn_embedding_out.append(x)

        cnn_embedding_out = torch.stack(cnn_embedding_out, dim=0).transpose(0, 1)

        self.rnn.flatten_parameters()
        rnn_out, _ = self.rnn(cnn_embedding_out, None)

        if self.bi_branch:
            for t in range(rnn_out.size(1)):
                x = rnn_out[:, t, :]
                x = self.fc_cnn(x)
                cnn_pred.append(x)
            cnn_pred = torch.stack(cnn_pred, dim=0).transpose(0, 1)

        x = self.global_pool(rnn_out)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_rnn(x)

        if self.inputgate:
            if self.bi_branch:
                return x, cnn_pred.reshape(-1, self.num_classes), torch.stack(gates, dim=0).view(-1, 192, 1)
            else:
                return x, gates
        else:
            if self.bi_branch:
                return x, cnn_pred.reshape(-1, self.num_classes)
            else:
                return x

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if 'inp_gate_l' in str(name):
                m.weight.data.normal_(0, 0.001)
                m.bias.data[::2].fill_(0.1)
                m.bias.data[1::2].fill_(2)
            elif 'inp_gate' in str(name):
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)


class CNN(nn.Module):
    def __init__(self, bi_branch=False, num_classes=2):
        super(CNN, self).__init__()

        self.num_classes = num_classes

        # 使用resnet预训练模型来提取特征，去掉最后一层分类器
        pretrained_cnn = Models.resnet50(pretrained=True)
        cnn_layers = list(pretrained_cnn.children())[:-1]

        # 把resnet的最后一层fc层去掉，用来提取特征
        self.cnn = nn.Sequential(*cnn_layers)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.cnn_out = nn.Sequential(
            nn.Linear(2048, 2)
        )

    def forward(self, x_3d):
        """
        输入的是T帧图像，shape = (batch_size, t, h, w, 3)
        """
        cnn_embedding_out = []
        for t in range(x_3d.size(1)):
            # 使用cnn提取特征
            x = self.cnn(x_3d[:, t, :, :, :])
            x = torch.flatten(x, start_dim=1)
            x = self.cnn_out(x)
            cnn_embedding_out.append(x)
        cnn_embedding_out = torch.stack(cnn_embedding_out, dim=0).transpose(0, 1)

        x = self.global_pool(cnn_embedding_out)
        x = torch.flatten(x, start_dim=1)

        return x


class cRNN(nn.Module):
    def __init__(self, use_gru=False, bi_branch=False, num_classes=2):
        super(cRNN, self).__init__()

        self.num_classes = num_classes
        self.use_gru = use_gru

        # 使用resnet预训练模型来提取特征，去掉最后一层分类器
        pretrained_cnn = Models.resnet50(pretrained=True)
        cnn_layers = list(pretrained_cnn.children())[:-2]

        # 把resnet的最后一层fc层去掉，用来提取特征
        self.cnn = nn.Sequential(*cnn_layers)

        cRNN_params = {
            'input_dim': 2048,
            'hidden_dim': [256, 256, 512],
            'kernel_size': (1, 1),
            'num_layers': 3,
            'batch_first': True
        } if not use_gru else {
            'input_size': (2, 2),
            'input_dim': 2048,
            'hidden_dim': [256, 256, 512],
            'kernel_size': (1, 1),
            'num_layers': 3,
            'batch_first': True
        }

        self.cRNN = (ConvGRU if use_gru else ConvLSTM)(**cRNN_params)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Sequential(
            nn.Linear(512, self.num_classes)
        )

    def forward(self, x_3d):
        cnn_embedding_out = []
        for t in range(x_3d.size(1)):
            # 使用cnn提取特征
            x = self.cnn(x_3d[:, t, :, :, :])
            cnn_embedding_out.append(x)

        x = torch.stack(cnn_embedding_out, dim=0).transpose(0, 1)

        _, outputs = self.cRNN(x)
        x = outputs[0][0] if self.use_gru else outputs[0][1]

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def get_resnet_3d(num_classes=2, model_depth=10, shortcut_type='B', sample_size=112, sample_duration=16):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = resnet.resnet10(
            num_classes=num_classes,
            shortcut_type=shortcut_type,
            sample_size=sample_size,
            sample_duration=sample_duration)
    elif model_depth == 18:
        model = resnet.resnet18(
            num_classes=num_classes,
            shortcut_type=shortcut_type,
            sample_size=sample_size,
            sample_duration=sample_duration)
    elif model_depth == 34:
        model = resnet.resnet34(
            num_classes=num_classes,
            shortcut_type=shortcut_type,
            sample_size=sample_size,
            sample_duration=sample_duration)
    elif model_depth == 50:
        model = resnet.resnet50(
            num_classes=num_classes,
            shortcut_type=shortcut_type,
            sample_size=sample_size,
            sample_duration=sample_duration)
    elif model_depth == 101:
        model = resnet.resnet101(
            num_classes=num_classes,
            shortcut_type=shortcut_type,
            sample_size=sample_size,
            sample_duration=sample_duration)
    elif model_depth == 152:
        model = resnet.resnet152(
            num_classes=num_classes,
            shortcut_type=shortcut_type,
            sample_size=sample_size,
            sample_duration=sample_duration)
    else:
        model = resnet.resnet200(
            num_classes=num_classes,
            shortcut_type=shortcut_type,
            sample_size=sample_size,
            sample_duration=sample_duration)

    return model
