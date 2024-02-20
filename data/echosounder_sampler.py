import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data

import paths
from data.dataset import EchosounderData

class SampleEchosounderData128():
    """
    DO NOT ERASE

    Preprocessing
    Initial range -75 ~ 0
    when loading: 0 ~ 75
    when prividing: 0~ 2 (after EchosounderData)

    # year_tr = ['2014' - '2017'] : 2018 patches
    # year_te = ['2018-2019']: 718 patches
    """
    def __init__(self, args):
        self.args = args
        self.path_to_echograms = paths.path_to_echograms()

    def for_train(self):
        data_tr = torch.load(os.path.join(self.path_to_echograms, 'echo_data_tr_paper3.pt'))
        cls_label_tr = torch.load(os.path.join(self.path_to_echograms, 'echo_label_tr_paper3.pt'))  # class
        seg_label_tr = torch.load(os.path.join(self.path_to_echograms, 'echo_seg_tr_paper3.pt'))  # class
        mask_prior_tr = torch.load(os.path.join(self.path_to_echograms, 'echo_prior_pooled_4x_tr_paper3.pt'))
        dataset_train = EchosounderData(
            data_in=data_tr,
            cls_labels_in=cls_label_tr,
            seg_labels_in=seg_label_tr,
            mask_prior_in=mask_prior_tr,
        )

        return dataset_train

    def for_val(self):
        data_val = torch.load(os.path.join(self.path_to_echograms, 'val_data_solid_paper3.pt'))
        cls_label_val = torch.load(os.path.join(self.path_to_echograms, 'val_cls_labels_solid_paper3.pt'))  # class
        seg_label_val = torch.load(os.path.join(self.path_to_echograms, 'val_seg_labels_solid_paper3.pt'))

        dataset_val = EchosounderData(
            data_val,
            cls_label_val,
            seg_label_val
        )
        return dataset_val

    def for_test(self):
        data_te = torch.load(os.path.join(self.path_to_echograms, 'echo_data_te_paper3.pt'))
        cls_label_te = torch.load(os.path.join(self.path_to_echograms, 'echo_label_te_paper3.pt'))  # class
        seg_label_te = torch.load(os.path.join(self.path_to_echograms, 'echo_seg_te_paper3.pt'))  # class
        mask_prior_te = torch.load(os.path.join(self.path_to_echograms, 'echo_prior_pooled_4x_te_paper3.pt'))

        dataset_test = EchosounderData(
            data_in = data_te,
            cls_labels_in = cls_label_te,
            seg_labels_in = seg_label_te,
            mask_prior_in=mask_prior_te,
        )
        return dataset_test

    def for_mixed(self):
        data_mixed = torch.load(os.path.join(self.path_to_echograms, 'data_mixed_paper3.pt'))
        cls_label_mixed = torch.load(os.path.join(self.path_to_echograms, 'cls_labels_mixed_paper3.pt'))  # class
        seg_label_mixed = torch.load(os.path.join(self.path_to_echograms, 'seg_labels_mixed_paper3.pt'))

        dataset_mixed = EchosounderData(
            data_mixed,
            cls_label_mixed,
            seg_label_mixed,
        )
        return dataset_mixed


#######################################################

