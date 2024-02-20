import torch
import torch.nn as nn
from models.network_architecture import classifier, ClassifierMnist
from utils.general_utils import xavier_init
import torch.nn.functional as F

class ClassifierOnly(nn.Module):
    def __init__(self, **kwargs):
        super(ClassifierOnly, self).__init__()
        self.args = kwargs['args']
        self.device = torch.device("cuda" if self.args.cuda else "cpu")
        self.decode_cnn = classifier(bn=self.args.bn, in_depth=self.args.explainer_in_depth,
                                     num_category=self.args.nmb_category)

    def decoder(self, x):
        pred = self.decode_cnn(x)
        return pred

    def forward(self, x, prior=None):
        logit = self.decoder(x)
        Z_hat_dummy = torch.zeros_like(x)
        MP_loss_dummy = F.max_pool2d(Z_hat_dummy, 4)
        XT_loss_dummy = torch.zeros(self.args.explainer_out_depth, 1).to(
            self.device)
        return logit, Z_hat_dummy, XT_loss_dummy, MP_loss_dummy

    def weight_init(self):
        for m in self._modules:
            if m == 'encode_cnn':
                xavier_init(self._modules[m].features)

class ClassifierOnlyMnist(nn.Module):
    def __init__(self, **kwargs):
        super(ClassifierOnlyMnist, self).__init__()
        self.args = kwargs['args']
        self.device = torch.device("cuda" if self.args.cuda else "cpu")
        self.decode_cnn = ClassifierMnist()

    def decoder(self, x):
        pred = self.decode_cnn(x)
        return pred

    def forward(self, x, prior=None):
        logit = self.decoder(x)
        Z_hat_dummy = torch.zeros_like(x)
        MP_loss_dummy = F.max_pool2d(Z_hat_dummy, 4)
        XT_loss_dummy = torch.zeros(self.args.explainer_out_depth, 1).to(self.device)
        return logit, Z_hat_dummy, XT_loss_dummy, MP_loss_dummy

    def weight_init(self):
        for m in self._modules:
            if m == 'encode_cnn':
                xavier_init(self._modules[m].features)
