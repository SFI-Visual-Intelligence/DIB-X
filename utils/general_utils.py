import torch
import torch.nn as nn
import argparse
# import torch.optim as optim
# import torch.nn.functional as F
# import sklearn.metrics as skm
# import numpy as np
# import os
# from sklearn.preprocessing import label_binarize

"""
Created on Oct 2022
@author: Changkyu Choi
"""
class Flatten2D(nn.Module):
    def __init__(self):
        super(Flatten2D, self).__init__()
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

class Flatten3D(nn.Module):
    def __init__(self, explainer_out_depth):
        super(Flatten3D, self).__init__()
        self.explainer_out_depth = explainer_out_depth

    def forward(self, x):
        x = x.view(x.size(0), self.explainer_out_depth, -1)
        return x

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class SumMeter(object):
    """Computes and stores the average and current value
        For confusion matrix"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0 # not avg but sum; for simplicity

    def update(self, val, n=None):
        self.val = val
        self.avg += val


def xavier_init(ms):
    for m in ms :
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(m.weight,gain=nn.init.calculate_gain('relu'))
            m.bias.data.zero_()


def patch_splitter(in_tensor, no_split):
    if no_split == 1:
        return in_tensor
    else:
        split_input = torch.tensor_split(in_tensor, no_split, dim=-1)
        split_input = torch.stack(split_input, dim=0)
        split_input = torch.tensor_split(split_input, no_split, dim=-2)
        split_input = torch.stack(split_input, dim=0)
        dim_split = split_input.shape # (4, 4, 128, 4 ,32, 32)
        out_dim = torch.tensor(-1).unsqueeze(0) # ([-1])
        for i, d in enumerate(dim_split):
            if i >= 2:
                out_dim = torch.cat((out_dim, torch.tensor(d).unsqueeze(0)), 0) # [-1, 128, 4, 32, 32]
        return split_input.view(tuple(out_dim))


def cuda(tensor, is_cuda):
    '''
    Send the tensor to cuda

    Args:
        is_cuda: logical. True or False

    Credit: https://github.com/1Konny/VIB-pytorch
    '''

    if is_cuda:
        return tensor.cuda()

    else:
        return tensor

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y',
                     '1', 'True', 'Y', 'Yes',
                     'YES', 'YEs', 'ye'):
        return True

    elif v.lower() in ('no', 'false', 'f', 'n',
                       '0', 'False', 'N', 'NO',
                       'No'):
        return False

    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# def patch_glue(split_tensor, no_split):
#     dim = split_tensor.shape
#     out_dim = torch.tensor((no_split, no_split))
#     for d in dim[1:]:
#         out_dim = torch.cat((out_dim, torch.tensor(d).unsqueeze(0)), 0)
#     glued_tensor = split_tensor.view(tuple(out_dim))
#     for axis in [-2, -1]:
#         glued_tensor = torch.cat(tuple(glued_tensor), dim=axis)
#     return glued_tensor


# def roc_curve_macro(label_vec, y_score, n_classes=3):
#     # Binarize the output
#     y_test = label_binarize(label_vec, classes=[0, 1, 2])
#     # Compute ROC curve and ROC area for each class
#     fpr = dict()
#     tpr = dict()
#     roc_auc = dict()
#     for i in range(n_classes):
#         fpr[i], tpr[i], _ = skm.roc_curve(y_test[:, i], y_score[:, i])
#         roc_auc[i] = skm.auc(fpr[i], tpr[i])
#
#     # First aggregate all false positive rates
#     all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
#
#     # Then interpolate all ROC curves at this points
#     mean_tpr = np.zeros_like(all_fpr)
#     for i in range(n_classes):
#         mean_tpr += interp(all_fpr, fpr[i], tpr[i])
#
#     # Finally average it and compute AUC
#     mean_tpr /= n_classes
#
#     fpr["macro"] = all_fpr
#     tpr["macro"] = mean_tpr
#     roc_auc["macro"] = skm.auc(fpr["macro"], tpr["macro"])
#
#     macro_roc_auc_ovo = skm.roc_auc_score(y_test, y_score, average="macro", multi_class="ovo")
#     weighted_roc_auc_ovo = skm.roc_auc_score(y_test, y_score, multi_class="ovo", average="weighted")
#     macro_roc_auc_ovr = skm.roc_auc_score(y_test, y_score, multi_class="ovr", average="macro")
#     weighted_roc_auc_ovr = skm.roc_auc_score(y_test, y_score, multi_class="ovr", average="weighted")
#     print("One-vs-One ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
#           "(weighted by prevalence)"
#           .format(macro_roc_auc_ovo, weighted_roc_auc_ovo))
#     print("One-vs-Rest ROC AUC scores:\n{:.6f} (macro),\n{:.6f} "
#           "(weighted by prevalence)"
#           .format(macro_roc_auc_ovr, weighted_roc_auc_ovr))
#
#     roc_auc_macro = {'macro_ovo': macro_roc_auc_ovo,
#                      'weighted_ovo':weighted_roc_auc_ovo,
#                      'macro_ovr': macro_roc_auc_ovr,
#                      'weighted_ovr': weighted_roc_auc_ovr}
#     return fpr, tpr, roc_auc, roc_auc_macro
#
# def conf_mat(ylabel, ypred, args):
#     mat = np.zeros([args.n_classes, args.n_classes])
#     # gt: axis 0, pred: axis 1
#     for (gt, pd) in zip(ylabel, ypred):
#         mat[gt, pd] += 1
#     f1_score = skm.f1_score(ylabel, ypred, average=args.f1_avg)
#     kappa = skm.cohen_kappa_score(ylabel, ypred)
#     prob_mat = np.zeros([args.n_classes, args.n_classes])
#     # gt: axis 0, pred: axis 1
#     for i in range(args.n_classes):
#         prob_mat[i] = mat[i] / np.sum(mat[i])
#     return prob_mat, mat, f1_score, kappa

