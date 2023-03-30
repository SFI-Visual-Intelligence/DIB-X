import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from pathlib import Path
from torchmetrics import F1Score, AUROC, CohenKappa, Accuracy, ConfusionMatrix, JaccardIndex
import wandb

from log_history import LogHistory
from data.return_data import return_data
from utils.general_utils import cuda
from utils.visualize import ExplainVisual, LogPlot


from models.dibx import DibxNetwork, DibxMnistNetwork, DibxMnistNetworkChunk1
from models.classifier_only import ClassifierOnly, ClassifierOnlyMnist
from models.vibi import VibiNetwork, VibiMnistNetwork
import cv2

# from models.gradcam import GradcamNetwork
# from models.lime import LimeNetwork

# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Solver(object):
    def __init__(self, data_type, args):
        self.data_type = data_type
        self.args = args
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self.class_criterion = nn.CrossEntropyLoss()

        # Dataset
        self.data_loader = return_data(args = self.args)
        self.x_type = self.data_loader['x_type']
        self.y_type = self.data_loader['y_type']

        # Network
        network_dic = {
            'onlyclassifier': ClassifierOnly,
            'dibx': DibxNetwork,
            'vibi': VibiNetwork,
            'onlyclassifiermnist': ClassifierOnlyMnist,
            'dibxmnist': DibxMnistNetwork,
            'dibxmnist-chunk1': DibxMnistNetworkChunk1,
            'vibimnist': VibiMnistNetwork,
            # 'gradcam': GradcamNetwork,
            # 'lime': LimeNetwork,
        }
        self.net = cuda(network_dic[self.args.model](args = self.args), self.args.cuda)
        self.net.weight_init() # initialize encode_cnn4. alexnet is automatically initialized when it is loaded
        self.optim = optim.SGD(self.net.parameters(), lr=self.args.lr_exp,
                              momentum=0.9, weight_decay=5e-4)
        self.optim_pretrain = optim.SGD(self.net.decode_cnn.parameters(), lr=self.args.lr_exp,
                              momentum=0.9, weight_decay=5e-4)
        self.args.global_epoch = -1

        # Load Checkpoint and History
        self.checkpoint_dir = Path(args.default_dir).joinpath(args.checkpoint_dir)
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.load_checkpoint_function(os.path.join(self.checkpoint_dir, 'current_acc.tar'))

        # seg_label: only for visualization
        self.image_dir_train = Path(self.args.default_dir).joinpath(self.args.checkpoint_dir, 'sample', 'train')
        if not self.image_dir_train.exists(): self.image_dir_train.mkdir(parents=True, exist_ok=True)

        self.image_dir_val = Path(self.args.default_dir).joinpath(self.args.checkpoint_dir, 'sample', data_type)
        if not self.image_dir_val.exists(): self.image_dir_val.mkdir(parents=True, exist_ok=True)

        self.image_dir_best = Path(self.args.default_dir).joinpath(self.args.checkpoint_dir, 'sample', 'best')
        if not self.image_dir_best.exists(): self.image_dir_best.mkdir(parents=True, exist_ok=True)

        self.history = LogHistory(save_accuracy=self.args.save_accuracy) # save checkpoint for higher accu than one in the the args

        # metrics
        self.metrics_dic = {
            'auroc': AUROC(task='multiclass', num_classes=self.args.nmb_category, average=None).to(self.device),
            'iou': JaccardIndex(task='multiclass', num_classes=self.args.nmb_category, average=None).to(self.device),
            'confusion': ConfusionMatrix(task='multiclass', num_classes=self.args.nmb_category, normalize='none').to(self.device),
            'accu': Accuracy(task='multiclass', num_classes=self.args.nmb_category, top_k=1).to(self.device),
            'f1': F1Score(task='multiclass', num_classes=self.args.nmb_category, top_k=1, average=None).to(self.device),
            'kappa': CohenKappa(task='multiclass', num_classes=self.args.nmb_category).to(self.device),
        }

    @staticmethod
    def combine_explain_and_pred(prediction, pred_activ):
        for i, (c, p) in enumerate(zip(prediction, pred_activ)):
            if i == 0:
                explain_seg = torch.mul(c, p).unsqueeze(0)
            else:
                explain_seg = torch.cat([explain_seg, torch.mul(c, p).unsqueeze(0)], dim=0)
        return explain_seg

    def get_metric(self, logit_proba, y_class, encoded_mask, y_seg):
        with torch.no_grad():
            for k in self.metrics_dic.keys():
                if k == 'iou':
                    if y_seg is not None:
                        prediction = logit_proba.argmax(-1)
                        pred_activ = (encoded_mask >= 0.5).float()
                        exp_seg = self.combine_explain_and_pred(prediction, pred_activ)
                        exp_seg = exp_seg.to(self.device)
                        yseg_expand = y_seg.unsqueeze(1).tile(1, 4, 1, 1).to(self.device)

                        iou_per_class = self.metrics_dic[k](exp_seg, yseg_expand)
                        for c, iouc in enumerate(iou_per_class):
                            iou_avg_key = 'iou_avg_c%d' %c
                            self.history.avg[iou_avg_key].update(iouc.item(), len(y_class))

                        if self.args.explainer_out_depth > 1:
                            exp_seg = exp_seg.transpose(0, 1)  # for iou per each channel
                            for f, (exp_seg_f) in enumerate(exp_seg):
                                iou_per_class_freq = self.metrics_dic[k](exp_seg_f, y_seg)
                                for cf, ioucf in enumerate(iou_per_class_freq):
                                    iou_cf_key = 'iou_f%d_c%d' % (f, cf)
                                    self.history.avg[iou_cf_key].update(ioucf.item(), len(y_class))
                elif k == 'confusion':
                    self.history.avg[k].update(self.metrics_dic[k](logit_proba, y_class), len(y_class))
                elif k == 'auroc':
                    auroc_per_class = self.metrics_dic[k](logit_proba, y_class)
                    self.history.avg['auroc_avg'].update(auroc_per_class.mean().item(), len(y_class))
                    for c, auroccf in enumerate(auroc_per_class):
                        auroc_key = 'auroc_c%d' % c
                        self.history.avg[auroc_key].update(auroccf.item(), len(y_class))
                elif k == 'f1':
                    f1_per_class = self.metrics_dic[k](logit_proba, y_class)
                    self.history.avg['f1_macro'].update(f1_per_class.mean().item(), len(y_class))
                    for c, f1cf in enumerate(f1_per_class):
                        f1_key = 'f1_c%d' % c
                        self.history.avg[f1_key].update(f1cf.item(), len(y_class))
                else:
                    self.history.avg[k].update(self.metrics_dic[k](logit_proba, y_class).item(), len(y_class))

    def set_mode(self, mode='train'):
        if mode == 'train':
            self.net.train()
        elif mode == 'eval':
            self.net.eval()
        else : raise('mode error. It should be either train or eval')

    def save_checkpoint_function(self, filename):
        history_save = {
            'best': self.history.best,
            'tr': self.history.tr_hist,
            'te': self.history.te_hist,
        }
        model_states = {
                'net':self.net.state_dict(),
                }
        optim_states = {
                'optim':self.optim.state_dict(),
                }
        states = {
                'iter':self.global_iter,
                'epoch':self.args.global_epoch,
                'history': history_save,
                'args':self.args,
                'model_states':model_states,
                'optim_states':optim_states,
                }
        torch.save(states, open(str(filename), 'wb+'))
        print("=> saved checkpoint '{}' (epoch {} iter {})".format(filename, self.args.global_epoch, self.global_iter))

    def load_checkpoint_function(self, filename):
        self.history = LogHistory(save_accuracy=self.args.save_accuracy)
        if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(open(str(filename), 'rb'), map_location=self.device)
            self.args.global_epoch = checkpoint['epoch']
            self.global_iter = checkpoint['iter']

            history_save = checkpoint['history']
            for k in history_save.keys():
                if k == 'best':
                    self.history.best = history_save[k]
                elif k == 'tr':
                    self.history.tr_hist = history_save[k]
                elif k == 'te':
                    self.history.te_hist = history_save[k]
            if 'best' not in history_save.keys():
                self.history.check_best_result()

            self.net.load_state_dict(checkpoint['model_states']['net'])
            print("=> loaded checkpoint '{} (epoch {} iter {})'".format(
                filename, self.args.global_epoch, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(filename))
            self.args.global_epoch = -1
            self.global_iter = 0

    def early_stopping(self, epoch_patience=100, interval=20, observe_key='total_loss'):
        '''
        tr_loss trend: keep decreasing over the epoch_patience
        te_loss trend: keep increasing over the epoch_patience
        '''
        earlystop = False
        if len(self.history.te_hist[observe_key]) > epoch_patience:
            tr_loss_epoch = torch.tensor(self.history.tr_hist[observe_key][-epoch_patience:])
            te_loss_epoch = torch.tensor(self.history.te_hist[observe_key][-epoch_patience:])
            tr_diff = tr_loss_epoch[interval:] - tr_loss_epoch[:-interval]
            te_diff = te_loss_epoch[interval:] - te_loss_epoch[:-interval]
            if (tr_diff.mean() < 0) and (te_diff.mean() > 0):
                print("Early stopping at epoch: ", self.args.global_epoch, ', iter:', self.global_iter)
                earlystop = True
                return earlystop
        else:
            return earlystop

    def pretrain(self, tr_data_type='train'): # pretrain classifier (decode_cnn)
        if self.args.global_epoch < 1:
            self.set_mode('train')
            with torch.autograd.set_detect_anomaly(True):
                for e in range(self.args.pretrain_epoch):
                    self.history.avg_reset()
                    for idx, batch in enumerate(self.data_loader[tr_data_type]):
                        x_raw = batch[0]
                        y_raw = batch[1]
                        x = Variable(cuda(x_raw, self.args.cuda)).type(self.x_type)
                        y = Variable(cuda(y_raw, self.args.cuda)).type(self.y_type)
                        logit = self.net.decode_cnn(x)
                        y_class = y if len(y.size()) == 1 else torch.argmax(y, dim=-1)
                        if self.args.model[:4] == 'vibi':  # logit is the output of the log-softmax
                            TY_loss_tr_pretrain = self.class_criterion(logit, y_class).div(torch.log(torch.tensor(2)))
                        else: # logit is the output of the linear layer (no softmax applied yet - directly processed at the self.class_criterion, (cross-entropy)
                            TY_loss_tr_pretrain = self.class_criterion(logit, y_class)

                        self.optim_pretrain.zero_grad()
                        TY_loss_tr_pretrain.backward()
                        self.optim_pretrain.step()

                        with torch.no_grad():
                            logit_proba = F.softmax(logit, dim=-1)
                            self.history.avg['TY_loss_pretrain'].update(TY_loss_tr_pretrain.item(), len(y_class))
                            self.history.avg['accu_pretrain'].update(self.metrics_dic['accu'](logit_proba, y_class).item(), len(y_class))

                    wandb.log({"TY_loss_pretrain": self.history.avg['TY_loss_pretrain'].avg,
                               "accu_pretrain": self.history.avg['accu_pretrain'].avg
                               })
                    if self.history.avg['accu_pretrain'].avg > self.args.pretrain_accu:
                        break

    def train(self, tr_data_type='train'):
        self.set_mode('train')
        with torch.autograd.set_detect_anomaly(True):
            for e in range(self.args.epoch):
                # logit_save = []
                # encoded_mask_save = []

                self.args.global_epoch += 1
                for idx, batch in enumerate(self.data_loader[tr_data_type]):
                    self.global_iter += 1

                    if self.args.data =='echograms':
                        x_raw = batch[0]
                        y_raw = batch[1]
                        if self.args.model[:4] == 'only': #onlyclassifier
                            y_seg = None
                        else:
                            y_seg = batch[2]
                            y_seg = Variable(cuda(y_seg, self.args.cuda)).type(self.x_type)
                        m_prior = batch[3]
                        x = Variable(cuda(x_raw, self.args.cuda)).type(self.x_type)
                        y = Variable(cuda(y_raw, self.args.cuda)).type(self.y_type)
                        m_prior = Variable(cuda(m_prior, self.args.cuda)).type(self.x_type)
                        if (self.args.chunk_size > 4) and (m_prior.shape[-1] == 32):  # need more coarse prior
                            scale = self.args.chunk_size // 4
                            m_prior = F.max_pool2d(m_prior, scale, scale)
                        logit, encoded_mask, XT_loss_tr, MP_loss_tr = self.net(x, prior=m_prior)

                    else:
                        x_raw = batch[0]
                        y_raw = batch[1]
                        y_seg = None
                        x = Variable(cuda(x_raw, self.args.cuda)).type(self.x_type)
                        y = Variable(cuda(y_raw, self.args.cuda)).type(self.y_type)
                        logit, encoded_mask, XT_loss_tr, MP_loss_tr = self.net(x)

                    y_class = y if len(y.size()) == 1 else torch.argmax(y, dim=-1)
                    if self.args.model[:4] == 'vibi': # logit is the output of the log-softmax
                        TY_loss_tr = self.class_criterion(logit, y_class).div(torch.log(torch.tensor(2)))
                    else: # logit is the output of the linear layer (no softmax applied yet - directly processed at the self.class_criterion, (cross-entropy)
                        TY_loss_tr = self.class_criterion(logit, y_class)
                    total_loss_tr = TY_loss_tr + XT_loss_tr.mean() + MP_loss_tr.mean()

                    self.optim.zero_grad()
                    total_loss_tr.backward()
                    self.optim.step()

                    '''
                    save metrics
                    XT_loss: patch_level (channel * MI_split * MI_split)
                    MP_loss : pixel level (shrinked by chunk size) 
                    '''
                    with torch.no_grad():
                        logit_proba = F.softmax(logit, dim=-1)
                        self.history.avg['TY_loss'].update(TY_loss_tr.item(), len(y_class))
                        self.history.avg['total_loss'].update(total_loss_tr.item(), len(y_class))
                        self.history.avg['XT_loss_avg'].update(XT_loss_tr.mean().item(), len(y_class))
                        self.history.avg['MP_loss_avg'].update(MP_loss_tr.mean().item(), len(y_class))
                        if self.args.explainer_out_depth > 1:
                            xt_f = XT_loss_tr.mean(-1)
                            mp_f = MP_loss_tr.mean((0, 2, 3))
                            for f in range(self.args.explainer_out_depth):
                                xtf_key = 'XT_loss_f%d' %f
                                mpf_key = 'MP_loss_f%d' %f
                                self.history.avg[xtf_key].update(xt_f[f].item(), len(y_class))
                                self.history.avg[mpf_key].update(mp_f[f].item(), len(y_class))
                        self.get_metric(logit_proba, y_class, encoded_mask, y_seg)
                        # logit_save.append(logit.detach().cpu())
                        # encoded_mask_save.append(encoded_mask.detach().cpu())

                with torch.no_grad():
                    self.history.tr_hist['epoch'].append(self.args.global_epoch)
                    self.history.tr_transfer()
                    update_flag = self.evaluation(data_type=self.data_type)

                    # earlystop
                    earlystop = self.early_stopping()

                    if earlystop:
                        print("EARLY STOPPING at epoch %d" % self.args.global_epoch)
                        break

    def visualization(self):
        self.set_mode('eval')
        if self.args.model[:4] != 'only':
            # visualize 3 mini-batches (test_set)
            with torch.no_grad():
                for idx_v, batch in enumerate(self.data_loader['visual']):
                    print(idx_v)
                    if idx_v > 0:
                        break
                    if self.args.data =='echograms':
                        x_raw = batch[0]
                        y_raw = batch[1]
                        y_seg = batch[2]
                        m_prior = batch[3]
                        x = Variable(cuda(x_raw, self.args.cuda)).type(self.x_type)
                        y_seg = Variable(cuda(y_seg, self.args.cuda)).type(self.x_type)
                        if (self.args.chunk_size > 4) and (m_prior.shape[-1] == 32):  # need more coarse prior
                            scale = self.args.chunk_size // 4
                            m_prior = F.max_pool2d(m_prior, scale, scale)

                    else:
                        x_raw = batch[0]
                        y_raw = batch[1]
                        y_seg = None
                        m_prior = None
                        x = Variable(cuda(x_raw, self.args.cuda)).type(self.x_type)
                    logit_visual, encoded_mask_visual, _, _ = self.net(x, visual=True)

                    ExplainVisual(image_dir=self.image_dir_best,
                                  idx=idx_v,
                                  data=self.args.data,
                                  epoch=self.args.global_epoch,
                                  batch=x_raw,
                                  label=y_raw,
                                  label_approx=logit_visual.argmax(-1),
                                  encoded_mask=encoded_mask_visual,
                                  seg_label=y_seg,
                                  prior=m_prior,
                                  threshold=self.args.saliency_visual_threshold
                                  )

    def evaluation(self, data_type): # data_type: 'valid', 'test', 'mixed'
        self.set_mode('eval')
        with torch.no_grad():
            # logit_save = []
            # encoded_mask_save = []

            for idx, batch in enumerate(self.data_loader[data_type]):
                if self.args.data =='echograms':
                    x_raw = batch[0]
                    y_raw = batch[1]
                    if self.args.model[:4] == 'only':
                        y_seg = None
                    else:
                        y_seg = batch[2]
                        y_seg = Variable(cuda(y_seg, self.args.cuda)).type(self.x_type)
                    m_prior = batch[3]
                    x = Variable(cuda(x_raw, self.args.cuda)).type(self.x_type)
                    y = Variable(cuda(y_raw, self.args.cuda)).type(self.y_type)
                    m_prior = Variable(cuda(m_prior, self.args.cuda)).type(self.x_type)
                    if (self.args.chunk_size > 4) and (m_prior.shape[-1] == 32): # need more coarse prior
                        scale = self.args.chunk_size//4
                        m_prior = F.max_pool2d(m_prior, scale, scale)
                    logit, encoded_mask, XT_loss_eval, MP_loss_eval = self.net(x, m_prior)

                else:
                    x_raw = batch[0]
                    y_raw = batch[1]
                    y_seg = None
                    x = Variable(cuda(x_raw, self.args.cuda)).type(self.x_type)
                    y = Variable(cuda(y_raw, self.args.cuda)).type(self.y_type)
                    logit, encoded_mask, XT_loss_eval, MP_loss_eval = self.net(x)

                y_class = y if len(y.size()) == 1 else torch.argmax(y, dim=-1)
                if self.args.model[:4] == 'vibi':  # logit is the output of the log-softmax
                    TY_loss_eval = self.class_criterion(logit, y_class).div(torch.log(torch.tensor(2)))
                else: # logit is the output of the linear layer (no softmax applied yet - directly processed at the self.class_criterion, (cross-entropy)
                    TY_loss_eval = self.class_criterion(logit, y_class)
                total_loss_eval = TY_loss_eval + XT_loss_eval.mean() + MP_loss_eval.mean()

                # logit_save.append(logit.detach().cpu())
                # encoded_mask_save.append(encoded_mask.detach().cpu())

                '''
                save metrics
                XT_loss: patch_level (channel * MI_split * MI_split)
                MP_loss : pixel level (shrinked by chunk size) 
                '''
                logit_proba = F.softmax(logit, dim=-1)
                self.history.avg['TY_loss'].update(TY_loss_eval.item(), len(y_class))
                self.history.avg['total_loss'].update(total_loss_eval.item(), len(y_class))
                self.history.avg['XT_loss_avg'].update(XT_loss_eval.mean().item(), len(y_class))
                self.history.avg['MP_loss_avg'].update(MP_loss_eval.mean().item(), len(y_class))
                if self.args.explainer_out_depth > 1:
                    xt_f = XT_loss_eval.mean(-1)
                    mp_f = MP_loss_eval.mean((0, 2, 3))
                    for f in range(self.args.explainer_out_depth):
                        xtf_key = 'XT_loss_f%d' %f
                        mpf_key = 'MP_loss_f%d' %f
                        self.history.avg[xtf_key].update(xt_f[f].item(), len(y_class))
                        self.history.avg[mpf_key].update(mp_f[f].item(), len(y_class))
                self.get_metric(logit_proba, y_class, encoded_mask, y_seg)

            self.history.te_hist['epoch'].append(self.args.global_epoch)
            update_flag = self.history.te_transfer()
            LogPlot(path=Path(self.args.default_dir),
                    data=self.args.data,
                    tr_hist=self.history.tr_hist,
                    te_hist=self.history.te_hist,
                    )

            if self.args.model[:4] == 'dibx':
                wandb.log({
                           "accu": self.history.te_hist['accu'][-1],
                           "best_accu": self.history.best['accu'],
                           "total_loss": self.history.te_hist['total_loss'][-1],
                           "best_total_loss": self.history.best['total_loss'],
                           "TY_loss": self.history.te_hist['TY_loss'][-1],
                           "XT_loss_avg": self.history.te_hist['XT_loss_avg'][-1],
                           "MP_loss_avg": self.history.te_hist['MP_loss_avg'][-1],
                           "dibx_mi_split": self.args.dibx_MI_patch_split,
                           "dibx_kernel": self.args.dibx_kernel_size,
                           "dibx_gamma": self.args.dibx_gamma,
                           "dibx_beta": self.args.dibx_beta,
                           "epoch": self.args.global_epoch,
                           })

            elif self.args.model[:4] == 'vibi':
                wandb.log({
                           "accu": self.history.te_hist['accu'][-1],
                           "best_accu": self.history.best['accu'],
                           "total_loss": self.history.te_hist['total_loss'][-1],
                           "best_total_loss": self.history.best['total_loss'],
                           "TY_loss": self.history.te_hist['TY_loss'][-1],
                           "XT_loss_avg": self.history.te_hist['XT_loss_avg'][-1],
                           "MP_loss_avg": self.history.te_hist['MP_loss_avg'][-1],
                           "vibi_gamma": self.args.vibi_gamma,
                           "vibi_beta": self.args.vibi_beta,
                           "vibi_tau": self.args.vibi_tau,
                           "vibi_num_avg_sample": self.args.vibi_num_avg_sample,
                           "vibi_K": self.args.vibi_K,
                           "epoch": self.args.global_epoch,
                           })
            elif self.args.model[:4] == 'only':
                wandb.log({
                           "accu": self.history.te_hist['accu'][-1],
                           "epoch": self.args.global_epoch,
                           "best_accu": self.history.best['accu'],
                           "total_loss": self.history.te_hist['total_loss'][-1],
                           "best_total_loss": self.history.best['total_loss'],
                           "TY_loss": self.history.te_hist['TY_loss'][-1],
                           "MP_loss_avg": self.history.te_hist['MP_loss_avg'][-1],
                           })

            # save and visualization
            if update_flag:
                self.save_checkpoint_function(os.path.join(self.checkpoint_dir, 'current_acc.tar'))
                self.save_checkpoint_function(os.path.join(self.checkpoint_dir, '%d_current_acc.tar' % self.args.global_epoch))

                if self.args.model[:4] != 'only':
                    # visualize 3 mini-batches (test_set)
                    with torch.no_grad():
                        for idx_v, batch in enumerate(self.data_loader['visual']):
                            if idx_v > 3:
                                break
                            if self.args.data =='echograms':
                                x_raw = batch[0]
                                y_raw = batch[1]
                                y_seg = batch[2]
                                m_prior = batch[3]
                                x = Variable(cuda(x_raw, self.args.cuda)).type(self.x_type)
                                y_seg = Variable(cuda(y_seg, self.args.cuda)).type(self.x_type)
                                if (self.args.chunk_size > 4) and (m_prior.shape[-1] == 32):  # need more coarse prior
                                    scale = self.args.chunk_size // 4
                                    m_prior = F.max_pool2d(m_prior, scale, scale)


                            else:
                                x_raw = batch[0]
                                y_raw = batch[1]
                                y_seg = None
                                m_prior = None
                                x = Variable(cuda(x_raw, self.args.cuda)).type(self.x_type)
                            logit_visual, encoded_mask_visual, _, _ = self.net(x, visual=True)

                            logit_and_mask = {'logit': logit_visual,
                                              'mask': encoded_mask_visual,
                                              'x_raw': x_raw,
                                              'y_raw': y_raw,
                                              'y_seg': y_seg,
                                              'm_prior': m_prior,
                                              'idx_v': idx_v,
                                              'data': self.args.data,
                                              'epoch': self.args.global_epoch,
                                              'model': self.args.model,
                                              }
                            torch.save(logit_and_mask, os.path.join(self.image_dir_val,
                                                                    '%d_%d_logit_n_mask_visual.pt' % (self.args.global_epoch, idx_v)))
                            ExplainVisual(image_dir=self.image_dir_best,
                                          idx=idx_v,
                                          data=self.args.data,
                                          epoch=self.args.global_epoch,
                                          batch=x_raw,
                                          label=y_raw,
                                          label_approx=logit_visual.argmax(-1),
                                          encoded_mask=encoded_mask_visual,
                                          seg_label=y_seg,
                                          prior=m_prior,
                                          threshold=self.args.saliency_visual_threshold
                                          )

        self.set_mode('train')
        return update_flag
