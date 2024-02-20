import torch
import os
from matplotlib.colors import ListedColormap
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
from pathlib import Path
import sys
current_dir = os.getcwd()
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..'))
from torchmetrics import JaccardIndex
from log_history import LogHistory


class ComputeIOU(object):
    def __init__(self, data_setup, **kwargs):
        self.figsize_scale = 2
        self.epoch = kwargs['epoch']
        self.idx = kwargs['idx']
        self.data_setup = data_setup
        self.filename = '%d_%d_logit_n_mask_visual.pt' % (self.epoch, self.idx)
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.load = torch.load(os.path.join(kwargs['image_dir'], self.data_setup, self.filename), map_location=self.device)
        self.data = self.load['data']  # seal or echo
        # self.threshold = kwargs['threshold']
        # self.image_dir = os.path.join(kwargs['image_dir'], self.data_setup, str(self.threshold))
        # if not os.path.isdir(self.image_dir):
        #     os.mkdir(self.image_dir)
        self.batch = self.load['x_raw']
        self.label = self.load['y_raw']
        self.label_approx = self.load['logit'].argmax(-1)
        self.encoded_mask = self.load['mask']
        for i in range(len(self.encoded_mask)):
            if (self.encoded_mask[i].max() > 0) and (self.encoded_mask[i].max() < 1):
                self.encoded_mask[i] = self.encoded_mask[i]/self.encoded_mask[i].max()

        self.chunk_size = 8
        if self.data == 'mnist':
            self.chunk_size = 4

        if self.encoded_mask.shape[-1] == self.batch.shape[-1]:
            self.encoded_mask = F.max_pool2d(self.encoded_mask, self.chunk_size, self.chunk_size)

        self.seg_label = self.load['y_seg']
        self.prior = self.load['m_prior']
        self.cmap = 'gist_heat_r'
        self.colormap_v_max_scale = 1.5

        self.rgb_colormap = ListedColormap([[1, 1, 1], [1, 1, 1], [1, 1, 1],
                                            [0, 0, 1], [0, 0, 1], [0, 0, 1],
                                            [1, 0, 0], [1, 0, 0], [1, 0, 0]])
        self.width = self.batch.size(-1)
        self.height = self.batch.size(-2)
        self.nmb_category = 3
        self.chunk_size = 8
        self.history = LogHistory(save_accuracy=0) # save checkpoint for higher accu than one in the the args
        self.iou_metric = JaccardIndex(task='multiclass', num_classes=self.nmb_category, average=None).to(self.device)

    @staticmethod
    def combine_explain_and_pred(prediction, pred_activ):
        for i, (c, p) in enumerate(zip(prediction, pred_activ)):
            if i == 0:
                explain_seg = torch.mul(c, p).unsqueeze(0)
            else:
                explain_seg = torch.cat([explain_seg, torch.mul(c, p).unsqueeze(0)], dim=0)
        return explain_seg

    def iou_metric(self):
        prediction = self.label_approx
        pred_activ = (self.encoded_mask >= 0.5).float()
        exp_seg = self.combine_explain_and_pred(prediction, pred_activ)
        exp_seg = F.interpolate(exp_seg, scale_factor=(self.chunk_size, self.chunk_size), mode='nearest')
        exp_seg = exp_seg.to(self.device)
        yseg_expand = self.seg_label.unsqueeze(1).tile(1, 4, 1, 1).to(self.device)

        iou_per_class = self.iou_metric(exp_seg, yseg_expand)
        for c, iouc in enumerate(iou_per_class):
            iou_avg_key = 'iou_avg_c%d' % c
            self.history.avg[iou_avg_key].update(iouc.item(), len(self.label))

        exp_seg = exp_seg.transpose(0, 1)  # for iou per each channel
        for f, (exp_seg_f) in enumerate(exp_seg):
            iou_per_class_freq = self.iou_metric(exp_seg_f, self.seg_label)
            for cf, ioucf in enumerate(iou_per_class_freq):
                iou_cf_key = 'iou_f%d_c%d' % (f, cf)
                self.history.avg[iou_cf_key].update(ioucf.item(), len(self.label))
        return iou_per_class