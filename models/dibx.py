import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import pdist, squareform

from models.network_architecture import explainer, classifier, ExplainerMnist, ClassifierMnist, ExplainerMnistChunk1
from utils.dibx_utils import calculate_MI
from utils.general_utils import patch_splitter, xavier_init


class DibxBase(nn.Module):
    def __init__(self):
        super(DibxBase, self).__init__()
        self.divergence_loss = nn.KLDivLoss(reduction="none")
        self.eps = 1e-7
        self.sigmat_min = 1e-2

    def encoder(self, x):
        Z_hat = self.encode_cnn(x)
        return Z_hat

    def decoder(self, masked_input_x):
        pred = self.decode_cnn(masked_input_x)
        return pred

    def MI_compute(self, X, T):
        with torch.no_grad():
            TT = copy.deepcopy(T.cpu().detach().numpy())
            KKt = squareform(
                pdist(TT.reshape(TT.shape[0], -1),
                      'euclidean'))  # Calculate Euclidiean distance between all samples.
            sigmat = np.mean(np.mean(np.sort(KKt, 1)[:, :self.args.dibx_kernel_size]))
            if sigmat < self.sigmat_min:
                sigmat = self.sigmat_min # to avoid nan result in calcuating MI
            XX = copy.deepcopy(X.cpu().detach().numpy())
            KKx = squareform(pdist(XX.reshape(XX.shape[0], -1),
                                   'euclidean'))  # Calculate Euclidiean distance between all samples.
            sigmax = np.mean(np.mean(np.sort(KKx, 1)[:, :self.args.dibx_kernel_size]))
        return calculate_MI(X, T, s_x=sigmax ** 2, s_y=sigmat ** 2)


    def forward(self, x, prior=None, visual=False):
        # introduce the prior of the mask
        Z_hat_small = self.encoder(x)  # probability of each element to be selected
        if self.chunk_size > 1:
            Z_hat = F.interpolate(Z_hat_small, scale_factor=(self.chunk_size, self.chunk_size), mode='nearest')
        else:
            Z_hat = Z_hat_small
        masked_input_x = torch.mul(x, Z_hat)

        # ############################################
        # Compute MI(X, T) by freq and patches for new sigma.
        # ############################################
        MP_loss = torch.zeros(Z_hat_small.shape)
        if prior is not None:
            MP_loss = self.divergence_loss(torch.log(Z_hat_small + self.eps), prior.to(self.device))  # pixel-level

        XT_loss = torch.zeros(self.args.explainer_out_depth, self.args.dibx_MI_patch_split * self.args.dibx_MI_patch_split).to(
            self.device)  # empty XT_loss

        if self.args.explainer_out_depth > 1: # multi-mask by f
            masked_input_x = torch.transpose(masked_input_x, 0, 1)  # transpose x
            x = torch.transpose(x, 0, 1)  # transpose x
        else:
            masked_input_x = masked_input_x.unsqueeze(0)
            x = x.unsqueeze(0)

        for f in range(self.args.explainer_out_depth):
            if self.args.dibx_MI_patch_split > 1:
                split_x = patch_splitter(x[f], self.args.dibx_MI_patch_split)
                split_masked_x = patch_splitter(masked_input_x[f], self.args.dibx_MI_patch_split)
            else: # no split, e.g. self.args.dibx_MI_patch_split == 1
                split_x = x[f].unsqueeze(0)
                split_masked_x = masked_input_x[f].unsqueeze(0)

            for i, (sp_x, sp_masked_x) in enumerate(zip(split_x, split_masked_x)):
                XT_loss[f][i] = self.MI_compute(X=sp_x, T=sp_masked_x)

        if self.args.explainer_out_depth > 1:  # multi-mask
            masked_input_x = torch.transpose(masked_input_x, 1, 0) # return transpose
        else:
            masked_input_x = masked_input_x.squeeze(0)

        logit = self.decoder(masked_input_x)
        # divergence loss
        if visual == True:
            return logit, Z_hat_small, XT_loss * self.args.dibx_beta, MP_loss * self.args.dibx_gamma
        else:
            return logit, Z_hat, XT_loss * self.args.dibx_beta, MP_loss * self.args.dibx_gamma

    def weight_init(self):
        for m in self._modules:
            if m == 'encode_cnn':
                xavier_init(self._modules[m].features)


class DibxMnistNetwork(DibxBase):
    def __init__(self, **kwargs):
        super(DibxMnistNetwork, self).__init__()
        self.args = kwargs['args']
        self.device = torch.device("cuda" if self.args.cuda else "cpu")
        self.chunk_size = self.args.chunk_size
        self.encode_cnn = ExplainerMnist()
        self.decode_cnn = ClassifierMnist()


################
class DibxNetwork(DibxBase):
    def __init__(self, **kwargs):
        super(DibxNetwork, self).__init__()
        self.args = kwargs['args']
        self.device = torch.device("cuda" if self.args.cuda else "cpu")
        self.chunk_size = self.args.chunk_size
        self.encode_cnn = explainer(bn=self.args.bn, method=self.args.model, in_depth=self.args.explainer_in_depth,
                                    out_depth=self.args.explainer_out_depth)
        self.decode_cnn = classifier(bn=self.args.bn, in_depth=self.args.explainer_in_depth,
                                     num_category=self.args.nmb_category)


class DibxMnistNetworkChunk1(DibxBase):
    def __init__(self, **kwargs):
        super(DibxMnistNetworkChunk1, self).__init__()
        self.args = kwargs['args']
        self.device = torch.device("cuda" if self.args.cuda else "cpu")
        self.chunk_size = self.args.chunk_size
        self.encode_cnn = ExplainerMnistChunk1()
        self.decode_cnn = ClassifierMnist()
