import numpy as np
import os
import torch
from utils.general_utils import AverageMeter, SumMeter

class LogHistory(object):
    def __init__(self, save_accuracy):
        self.save_accuracy = save_accuracy
        self.best = {'epoch': 0,
                     }
        self.tr_hist = {'epoch': [],
                        }
        self.te_hist = {'epoch': [],
                        }
        self.val_hist = {'epoch': [],
                         }
        self.avg = { 'TY_loss_pretrain': AverageMeter(),
                     'accu_pretrain': AverageMeter(),
                     'TY_loss': AverageMeter(),
                     'XT_loss_avg': AverageMeter(),
                     'XT_loss_f0': AverageMeter(),
                     'XT_loss_f1': AverageMeter(),
                     'XT_loss_f2': AverageMeter(),
                     'XT_loss_f3': AverageMeter(),
                     'MP_loss_avg': AverageMeter(),
                     'MP_loss_f0': AverageMeter(),
                     'MP_loss_f1': AverageMeter(),
                     'MP_loss_f2': AverageMeter(),
                     'MP_loss_f3': AverageMeter(),
                     'total_loss': AverageMeter(),
                     'accu': AverageMeter(),
                     'confusion': SumMeter(),          ################### Summeter instead of Averagementer
                     'f1_macro': AverageMeter(),
                     'f1_c0': AverageMeter(),
                     'f1_c1': AverageMeter(),
                     'f1_c2': AverageMeter(),
                     'f1_c3': AverageMeter(),
                     'f1_c4': AverageMeter(),
                     'f1_c5': AverageMeter(),
                     'f1_c6': AverageMeter(),
                     'f1_c7': AverageMeter(),
                     'f1_c8': AverageMeter(),
                     'f1_c9': AverageMeter(),
                     'auroc_avg': AverageMeter(),
                     'auroc_c0': AverageMeter(),
                     'auroc_c1': AverageMeter(),
                     'auroc_c2': AverageMeter(),
                     'auroc_c3': AverageMeter(),
                     'auroc_c4': AverageMeter(),
                     'auroc_c5': AverageMeter(),
                     'auroc_c6': AverageMeter(),
                     'auroc_c7': AverageMeter(),
                     'auroc_c8': AverageMeter(),
                     'auroc_c9': AverageMeter(),
                     'kappa': AverageMeter(),
                     'iou_avg_c0': AverageMeter(),
                     'iou_avg_c1': AverageMeter(),
                     'iou_avg_c2': AverageMeter(),
                     'iou_f0_c0': AverageMeter(),
                     'iou_f0_c1': AverageMeter(),
                     'iou_f0_c2': AverageMeter(),
                     'iou_f1_c0': AverageMeter(),
                     'iou_f1_c1': AverageMeter(),
                     'iou_f1_c2': AverageMeter(),
                     'iou_f2_c0': AverageMeter(),
                     'iou_f2_c1': AverageMeter(),
                     'iou_f2_c2': AverageMeter(),
                     'iou_f3_c0': AverageMeter(),
                     'iou_f3_c1': AverageMeter(),
                     'iou_f3_c2': AverageMeter(),
                    }

    def avg_reset(self):
        for k in self.avg.keys():
            self.avg[k].reset()

    def tr_transfer(self):
        for k in self.avg.keys():
            if k not in self.tr_hist.keys():
                self.tr_hist[k] = [self.avg[k].avg]
            else:
                self.tr_hist[k].append(self.avg[k].avg)
        self.avg_reset()

    def te_transfer(self):
        for k in self.avg.keys():
            if k not in self.te_hist.keys():
                self.te_hist[k] = [self.avg[k].avg]
            else:
                self.te_hist[k].append(self.avg[k].avg)
        update_flag = self.check_best_result()
        self.avg_reset()
        return update_flag

    def val_transfer(self):
        for k in self.avg.keys():
            if k not in self.val_hist.keys():
                self.val_hist[k] = [self.avg[k].avg]
            else:
                self.val_hist[k].append(self.avg[k].avg)
        self.avg_reset()

    def check_best_result(self):
        update_flag = False
        if len(self.te_hist['epoch']) > 0:
            if self.best['epoch'] == 0:
                max_loca = torch.FloatTensor(self.te_hist['accu']).argmax().item()
                for k in self.te_hist.keys():
                    self.best[k] = self.te_hist[k][max_loca]
            else:
                if (self.best['accu'] < self.te_hist['accu'][-1]) and (self.save_accuracy < self.te_hist['accu'][-1]):
                    update_flag = True
                    for k in self.te_hist.keys():
                        self.best[k] = self.te_hist[k][-1]
        return update_flag
