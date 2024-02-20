from __future__ import print_function

import torch.nn.functional as F
import torch.nn as nn
import math
import torchvision.transforms as T
from utils.general_utils import Flatten2D


class Explainer(nn.Module):
    def __init__(self, features): #, out_depth, feature_last_channel):
        super(Explainer, self).__init__()
        self.features = features
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            # print(y, m)
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

class Classifier(nn.Module):
    def __init__(self, features, num_classes):
        super(Classifier, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(Flatten2D(),
                                        nn.Dropout(0.5),
                                        nn.Linear(256 * 3 * 3, 1024),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(0.5),
                                        nn.Linear(1024, 512),
                                        nn.ReLU(inplace=True),
                                        )
        self.top_layer = nn.Sequential(
            nn.Linear(512, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = self.top_layer(x)
        return x

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def make_layers_features(cfg, additional_layer, input_dim, bn):
    layers = []
    in_channels = input_dim
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'drop':
            layers += [nn.Dropout2d(p=0.2)]
        elif v[-1] == 'conv':
            conv2d = nn.Conv2d(in_channels, v[0], kernel_size=v[1], stride=v[2], padding=v[3])
            if bn:
                layers += [conv2d, nn.BatchNorm2d(v[0]), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v[0]
        elif v[-1] == 'tconv':
            tconv2d = nn.ConvTranspose2d(in_channels, v[0], kernel_size=v[1], stride=v[2], padding=v[3])
            if bn:
                layers += [tconv2d, nn.BatchNorm2d(v[0]), nn.ReLU(inplace=True)]
            else:
                layers += [tconv2d, nn.ReLU(inplace=True)]
            in_channels = v[0]
    if len(additional_layer):
        for add_lay in additional_layer:
            if add_lay == '-1':
                layers = layers[:-1]
            else:
                layers.append(add_lay)
    return nn.Sequential(*layers)

def classifier(bn, in_depth, num_category):
    CFG = {
        '2012': [(96, 11, 4, 2, 'conv'), 'drop', 'M',
                 (256, 5, 1, 2, 'conv'),  'drop',  'M',
                 (384, 3, 1, 1, 'conv'), (384, 3, 1, 1, 'conv'), (256, 3, 1, 1, 'conv'),  'drop', 'M']
   }
    model = Classifier(make_layers_features(CFG['2012'], additional_layer=[], input_dim=in_depth, bn=bn), num_category)
    return model

def explainer(bn, method, in_depth, out_depth):
    CFG = {
        #  For patch size 8
        'dibx': [(32, 5, 1, 2, 'conv'), (32, 5, 1, 2, 'conv'), 'drop', 'M',  # 128 / 128 / 64
                      (64, 5, 1, 2, 'conv'), (64, 5, 1, 2, 'conv'), 'drop', 'M',  # 64 / 64 / 32
                      (128, 5, 1, 2, 'conv'), (128, 5, 1, 2, 'conv'), 'drop', 'M',  # 32 / 32 / 16
                      (128, 5, 1, 2, 'conv'), (128, 5, 1, 2, 'conv'), 'drop', (64, 14, 2, 0, 'tconv'),  # 16 / 16 / 44
                      (64, 5, 1, 0, 'conv'), (64, 5, 1, 0, 'conv'), (64, 5, 1, 0, 'conv'), 'drop', 'M', # 40 / 36 / 32 / 16
                      (16, 1, 1, 0, 'conv'), (out_depth, 1, 1, 0, 'conv'),
                 ],
        'vibi': [(32, 5, 1, 2, 'conv'), (32, 5, 1, 2, 'conv'), 'drop', 'M',  # 128 / 128 / 64
                 (64, 5, 1, 2, 'conv'), (64, 5, 1, 2, 'conv'), 'drop', 'M',  # 64 / 64 / 32
                 (128, 5, 1, 2, 'conv'), (128, 5, 1, 2, 'conv'), 'drop', 'M',  # 32 / 32 / 16
                 (128, 5, 1, 2, 'conv'), (128, 5, 1, 2, 'conv'), 'drop', (64, 14, 2, 0, 'tconv'),  # 16 / 16 / 44
                 (64, 5, 1, 0, 'conv'), (64, 5, 1, 0, 'conv'), (64, 5, 1, 0, 'conv'), 'drop', 'M', # 40 / 36 / 32 / 16
                 (16, 1, 1, 0, 'conv'), (out_depth, 1, 1, 0, 'conv'),
                 ],
    }
    '''
    For patch size 4        
    # 'dibx': [(32, 5, 1, 2, 'conv'), (32, 5, 1, 2, 'conv'), 'drop', 'M',  # 128 / 128 / 64
    #               (64, 5, 1, 2, 'conv'), (64, 5, 1, 2, 'conv'), 'drop', 'M',  # 64 / 64 / 32
    #               (128, 5, 1, 2, 'conv'), (128, 5, 1, 2, 'conv'), 'drop', 'M',  # 32 / 32 / 16
    #               (128, 5, 1, 2, 'conv'), (128, 5, 1, 2, 'conv'), 'drop', (64, 14, 2, 0, 'tconv'),  # 16 / 16 / 44
    #               (64, 5, 1, 0, 'conv'), (64, 5, 1, 0, 'conv'), (64, 5, 1, 0, 'conv'), # 40 / 36 / 32
    #               (16, 1, 1, 0, 'conv'), (out_depth, 1, 1, 0, 'conv'),
    #          ],
    # 'vibi': [(32, 5, 1, 2, 'conv'), (32, 5, 1, 2, 'conv'), 'drop', 'M',  # 128 / 128 / 64
    #          (64, 5, 1, 2, 'conv'), (64, 5, 1, 2, 'conv'), 'drop', 'M',  # 64 / 64 / 32
    #          (128, 5, 1, 2, 'conv'), (128, 5, 1, 2, 'conv'), 'drop', 'M',  # 32 / 32 / 16
    #          (128, 5, 1, 2, 'conv'), (128, 5, 1, 2, 'conv'), 'drop', (64, 14, 2, 0, 'tconv'),  # 16 / 16 / 44
    #          (64, 5, 1, 0, 'conv'), (64, 5, 1, 0, 'conv'), (64, 5, 1, 0, 'conv'),  # 40 / 36 / 32
    #          (16, 1, 1, 0, 'conv'), (out_depth, 1, 1, 0, 'conv'),
    #          ]'''

    add_list = {
        'dibx': [T.GaussianBlur(kernel_size=(3, 3), sigma=(0.5, 0.5)), nn.Tanh()],
        'vibi': [],
               }
    model = Explainer(features=make_layers_features(CFG[method],
                                                    additional_layer=add_list[method], input_dim=in_depth, bn=bn))
    return model


class ExplainerMnist(nn.Module):
    def __init__(self): #, out_depth, feature_last_channel):
        super(ExplainerMnist, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size = (5, 5), padding = 2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size = (2, 2)),
            nn.Conv2d(8, 16, kernel_size = (5, 5), padding = 2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size = (2, 2)),
            nn.Conv2d(16, 1, kernel_size = (1, 1)),
            nn.ReLU(True),
            nn.Tanh(),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            # print(y, m)
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()

class ExplainerMnistChunk1(nn.Module):
    def __init__(self): #, out_depth, feature_last_channel):
        super(ExplainerMnistChunk1, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size = (5, 5), padding = 2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size = (2, 2)),
            nn.Conv2d(8, 16, kernel_size = (5, 5), padding = 2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size = (2, 2)),
            nn.Conv2d(16, 32, kernel_size = (5, 5), padding = 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=6, stride=2),  # 18
            nn.Conv2d(32, 16, kernel_size=(5, 5)),                                         # 14
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 16, kernel_size=6, stride=2),                           # 32
            nn.Conv2d(16, 8, kernel_size=(5, 5)),                                          # 28
            nn.ReLU(True),
            nn.Conv2d(8, 1, kernel_size=(1, 1)),  # 14
            nn.ReLU(True),
            nn.Tanh(),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        return x

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            # print(y, m)
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()


class ClassifierMnist(nn.Module):
    def __init__(self) :
        super(ClassifierMnist, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(5, 5), padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(32, 64, kernel_size=(5, 5), padding=2),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=(2, 2)),
            Flatten2D(),
        )
        self.top_layer = nn.Sequential(
            nn.Linear(7 * 7 * 64, 10),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.top_layer(x)
        return x

    def _initialize_weights(self):
        for y, m in enumerate(self.modules()):
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                for i in range(m.out_channels):
                    m.weight.data[i].normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


'''
def transconv_size(H_in, kernel_size, stride, padding=0, dilation=1, output_padding=0):
    return (H_in-1)*stride - 2*padding + dilation*(kernel_size-1) + output_padding + 1
    
def conv_size(H_in, kernel_size, stride, padding, dilation=1):
    return (H_in + 2 * padding - dilation * (kernel_size -1) -1)/stride + 1
    
'''