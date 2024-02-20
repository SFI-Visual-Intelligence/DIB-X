import torch
import matplotlib.pyplot as plt
import copy
from pathlib import Path
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import cv2

class ExplainVisual(object):
    def __init__(self, image_dir, idx, data, epoch,
                 batch, encoded_mask,
                 label, label_approx,
                 seg_label=None, prior=None, threshold=0.5):

        self.figsize_scale = 2
        self.image_dir = image_dir
        self.idx = idx
        self.data = data  # seal or echo
        self.epoch = epoch
        self.batch = batch
        self.label = label
        self.label_approx = label_approx
        self.encoded_mask = encoded_mask
        for i in range(len(self.encoded_mask)):
            if (self.encoded_mask[i].max() > 0) and (self.encoded_mask[i].max() < 1):
                self.encoded_mask[i] = self.encoded_mask[i]/self.encoded_mask[i].max()
        self.seg_label = seg_label
        self.prior = prior
        self.cmap = 'gist_heat_r'
        self.colormap_v_max_scale = 1.5
        self.threshold = threshold
        self.rgb_colormap = ListedColormap([[1, 1, 1], [1, 1, 1], [1, 1, 1],
                                            [0, 0, 1], [0, 0, 1], [0, 0, 1],
                                            [0, 1, 0], [0, 1, 0], [0, 1, 0]])
        self.width = self.batch.size(-1)
        self.height = self.batch.size(-2)

        if self.data == 'echograms':
            self.output = self.visualize_echo()
        else:
            self.output = self.visualize_others()

    def visualize_echo(self):
        with torch.no_grad():
            self.num_frequencies = 4
            enc_size = self.encoded_mask.size()
            if len(enc_size) == 3:
                self.encoded_mask = self.encoded_mask.unsqueeze(1)
            self.encoded_mask = self.encoded_mask.expand(
                    enc_size[0], self.num_frequencies, enc_size[-2], enc_size[-1])
            img = copy.deepcopy(self.batch.detach().cpu())
            encoded_mask = copy.deepcopy(self.encoded_mask.detach().cpu())
            thresholded_mask = copy.deepcopy(self.encoded_mask.detach().cpu())
            thresholded_mask[thresholded_mask < self.threshold] = 0
            seg_label = copy.deepcopy(self.seg_label.detach().cpu())
            prior = copy.deepcopy(self.prior.detach().cpu())
            n_img = img.size(0)
            fig_per_img = 5
            n_row = 3
            n_col = n_img // n_row + 1
            one_line = n_col * fig_per_img

            # small mask, prior, data and seg
            for f in range(self.num_frequencies):
                fig = plt.figure(figsize=(n_col * self.figsize_scale, n_row * fig_per_img * self.figsize_scale))
                att_filename = Path(self.image_dir).joinpath(
                    'i' + str(self.idx) + '_e' + str(self.epoch) + '_f' + str(f) + '.png')
                for i in range(n_img):
                    k = i // n_col
                    l = i % n_col

                    plt.subplot(n_row * fig_per_img, n_col, k * one_line + l + 1)
                    plt.axis('off')
                    s_mask = encoded_mask[i][f]
                    s_mask = cv2.resize(s_mask.cpu().numpy(), (self.width, self.height))
                    plt.title('L{}, Ap{}, f{}, e{}'.format(self.label[i].cpu(), self.label_approx[i].cpu(), f, self.epoch))
                    plt.imshow(s_mask, vmin=0, vmax=self.colormap_v_max_scale, cmap=self.cmap)

                    plt.subplot(n_row * fig_per_img, n_col, k * one_line + l + 1 + n_col)
                    plt.axis('off')
                    th_mask = thresholded_mask[i][f]
                    th_mask = cv2.resize(th_mask.cpu().numpy(), (self.width, self.height))
                    # plt.title('L{}, Ap{}, f{}, e{}'.format(self.label[i].cpu(), self.label_approx[i].cpu(), f, self.epoch))
                    plt.imshow(th_mask, vmin=0, vmax=self.colormap_v_max_scale, cmap=self.cmap)

                    plt.subplot(n_row * fig_per_img, n_col, k*one_line + l + 1 + n_col*2)
                    plt.axis('off')
                    plt.imshow(img[i][f], vmin=0, vmax=2)  # 200 kHz
                    # plt.title('L{}, Ap{}, f{}, e{}'.format(self.label[i].cpu(), self.label_approx[i].cpu(), f, self.epoch))


                    plt.subplot(n_row * fig_per_img, n_col, k*one_line + l + 1 + n_col*3)
                    plt.axis('off')
                    seg_l = seg_label[i].float()
                    # plt.title('L{}, Ap{}, f{}, e{}'.format(self.label[i].cpu(), self.label_approx[i].cpu(), f, self.epoch))
                    plt.imshow(seg_l, vmin=0, vmax=2, cmap=self.rgb_colormap, interpolation='nearest')
                    plt.imshow(th_mask, vmin=0, vmax=self.colormap_v_max_scale, cmap=self.cmap, alpha=0.5)

                    plt.subplot(n_row * fig_per_img, n_col, k * one_line + l + 1 + n_col * 4)
                    plt.axis('off')
                    pri = prior[i][f]
                    pri = cv2.resize(pri.cpu().numpy(), (self.width, self.height))
                    # plt.title('L{}, Ap{}, f{}, e{}'.format(self.label[i].cpu(), self.label_approx[i].cpu(), f, self.epoch))
                    plt.imshow(pri, vmin=0, vmax=self.colormap_v_max_scale, cmap=self.cmap)

                fig.subplots_adjust(wspace=0.05, hspace=0.35)
                plt.tight_layout()
                fig.savefig(str(att_filename))
                plt.close()
        return None

    def visualize_others(self):
        with torch.no_grad():
            img = copy.deepcopy(self.batch.detach().cpu())
            encoded_mask = copy.deepcopy(self.encoded_mask.detach().cpu())
            thresholded_mask = copy.deepcopy(self.encoded_mask.detach().cpu())
            thresholded_mask[thresholded_mask < self.threshold] = 0
            n_img = img.size(0)
            fig_per_img = 4
            n_row = 4
            n_col = n_img // n_row + 1
            one_line = n_col * fig_per_img

            # small mask, prior, data and seg
            fig = plt.figure(figsize=(n_col * self.figsize_scale, n_row * fig_per_img * self.figsize_scale))
            att_filename = Path(self.image_dir).joinpath(
                'i' + str(self.idx) + '_e' + str(self.epoch) + '.png')
            for i in range(n_img):
                k = i // n_col
                l = i % n_col

                plt.subplot(n_row * fig_per_img, n_col, k * one_line + l + 1)
                plt.axis('off')
                s_mask = encoded_mask[i].squeeze()
                s_mask = cv2.resize(s_mask.cpu().numpy(), (self.width, self.height))
                plt.title('L{}, Ap{}, e{}'.format(self.label[i].cpu(), self.label_approx[i].cpu(), self.epoch))
                plt.imshow(s_mask, vmin=0, vmax=self.colormap_v_max_scale, cmap=self.cmap)

                plt.subplot(n_row * fig_per_img, n_col, k*one_line + l + 1 + 1*n_col)
                plt.axis('off')
                th_mask = thresholded_mask[i].squeeze()
                th_mask = cv2.resize(th_mask.cpu().numpy(), (self.width, self.height))
                plt.imshow(th_mask, vmin=0, vmax=self.colormap_v_max_scale, cmap=self.cmap)

                plt.subplot(n_row * fig_per_img, n_col, k*one_line + l + 1 + 2*n_col)
                plt.axis('off')
                plt.imshow(img[i].permute(1, 2, 0).squeeze())  # rgb

                plt.subplot(n_row * fig_per_img, n_col, k*one_line + l + 1 + 3*n_col)
                plt.axis('off')
                plt.imshow(th_mask, vmin=0, vmax=self.colormap_v_max_scale, cmap=self.cmap)
                plt.imshow(img[i].permute(1, 2, 0).squeeze(), alpha=0.5)  # rgb

            fig.subplots_adjust(wspace=0.05, hspace=0.35)
            plt.tight_layout()
            fig.savefig(str(att_filename))
            plt.close()
        return None

class LogPlot(object):
    def __init__(self, path, data, tr_hist, te_hist):
        self.path = path
        self.data = data
        self.tr_hist = tr_hist
        self.te_hist = te_hist
        self.fig_scale = 4
        if self.data == 'echograms':
            self.output = self.plot_echo()
        else:
            self.output = self.plot_seal()

    def plot_echo(self):
        ncol = 8
        nrow = 4
        fig = plt.figure(figsize=(ncol*self.fig_scale, nrow*self.fig_scale))

        plt.subplot(nrow, ncol, ncol*0+1)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['accu'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['accu'], 'r--', label='te')
        plt.legend()
        plt.title('Accu')

        plt.subplot(nrow, ncol, ncol*1+1)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['TY_loss'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['TY_loss'], 'r--', label='te')
        plt.legend()
        plt.title('TY_loss')

        plt.subplot(nrow, ncol, ncol*2+1)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['XT_loss_avg'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['XT_loss_avg'], 'r--', label='te')
        plt.legend()
        plt.title('XT_loss_avg')

        plt.subplot(nrow, ncol, ncol*3+1)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['total_loss'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['total_loss'], 'r--', label='te')
        plt.legend()
        plt.title('total_loss')

        ######################################################

        plt.subplot(nrow, ncol, ncol*0+2)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['XT_loss_f0'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['XT_loss_f0'], 'r--', label='te')
        plt.legend()
        plt.title('XT_loss_f0')

        plt.subplot(nrow, ncol, ncol*1+2)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['XT_loss_f1'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['XT_loss_f1'], 'r--', label='te')
        plt.legend()
        plt.title('XT_loss_f1')

        plt.subplot(nrow, ncol, ncol*2+2)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['XT_loss_f2'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['XT_loss_f2'], 'r--', label='te')
        plt.legend()
        plt.title('XT_loss_f2')

        plt.subplot(nrow, ncol, ncol*3+2)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['XT_loss_f3'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['XT_loss_f3'], 'r--', label='te')
        plt.legend()
        plt.title('XT_loss_f3')

        ######################################################

        plt.subplot(nrow, ncol, ncol*0+3)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['f1_c0'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['f1_c0'], 'r--', label='te')
        plt.legend()
        plt.title('f1_c0')

        plt.subplot(nrow, ncol, ncol*1+3)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['f1_c1'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['f1_c1'], 'r--', label='te')
        plt.legend()
        plt.title('f1_c1')

        plt.subplot(nrow, ncol, ncol*2+3)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['f1_c2'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['f1_c2'], 'r--', label='te')
        plt.legend()
        plt.title('f1_c2')


        plt.subplot(nrow, ncol, ncol*3+3)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['kappa'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['kappa'], 'r--', label='te')
        plt.legend()
        plt.title('kappa')


        ######################################################

        plt.subplot(nrow, ncol, ncol*0+4)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['auroc_c0'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['auroc_c0'], 'r--', label='te')
        plt.legend()
        plt.title('auroc_c0')

        plt.subplot(nrow, ncol, ncol*1+4)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['auroc_c1'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['auroc_c1'], 'r--', label='te')
        plt.legend()
        plt.title('auroc_c1')

        plt.subplot(nrow, ncol, ncol*2+4)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['auroc_c2'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['auroc_c2'], 'r--', label='te')
        plt.legend()
        plt.title('auroc_c2')

        plt.subplot(nrow, ncol, ncol*3+4)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['auroc_avg'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['auroc_avg'], 'r--', label='te')
        plt.legend()
        plt.title('auroc_avg')

        ######################################################

        plt.subplot(nrow, ncol, ncol*0+5)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['iou_avg_c0'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['iou_avg_c0'], 'r--', label='te')
        plt.legend()
        plt.title('iou_avg_c0')

        plt.subplot(nrow, ncol, ncol*1+5)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['iou_avg_c1'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['iou_avg_c1'], 'r--', label='te')
        plt.legend()
        plt.title('iou_avg_c1')

        plt.subplot(nrow, ncol, ncol*2+5)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['iou_avg_c2'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['iou_avg_c2'], 'r--', label='te')
        plt.legend()
        plt.title('iou_avg_c2')


        ######################################################

        plt.subplot(nrow, ncol, ncol*0+6)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['iou_f0_c0'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['iou_f0_c0'], 'r--', label='te')
        plt.legend()
        plt.title('iou_f0_c0')

        plt.subplot(nrow, ncol, ncol*1+6)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['iou_f1_c0'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['iou_f1_c0'], 'r--', label='te')
        plt.legend()
        plt.title('iou_f1_c0')

        plt.subplot(nrow, ncol, ncol*2+6)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['iou_f2_c0'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['iou_f2_c0'], 'r--', label='te')
        plt.legend()
        plt.title('iou_f2_c0')

        plt.subplot(nrow, ncol, ncol*3+6)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['iou_f3_c0'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['iou_f3_c0'], 'r--', label='te')
        plt.legend()
        plt.title('iou_f3_c0')

        ######################################################

        plt.subplot(nrow, ncol, ncol*0+7)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['iou_f0_c1'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['iou_f0_c1'], 'r--', label='te')
        plt.legend()
        plt.title('iou_f0_c1')

        plt.subplot(nrow, ncol, ncol*1+7)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['iou_f1_c1'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['iou_f1_c1'], 'r--', label='te')
        plt.legend()
        plt.title('iou_f1_c1')

        plt.subplot(nrow, ncol, ncol*2+7)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['iou_f2_c1'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['iou_f2_c1'], 'r--', label='te')
        plt.legend()
        plt.title('iou_f2_c1')

        plt.subplot(nrow, ncol, ncol*3+7)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['iou_f3_c1'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['iou_f3_c1'], 'r--', label='te')
        plt.legend()
        plt.title('iou_f3_c1')

        ######################################################

        plt.subplot(nrow, ncol, ncol*0+8)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['iou_f0_c2'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['iou_f0_c2'], 'r--', label='te')
        plt.legend()
        plt.title('iou_f0_c2')

        plt.subplot(nrow, ncol, ncol*1+8)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['iou_f1_c2'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['iou_f1_c2'], 'r--', label='te')
        plt.legend()
        plt.title('iou_f1_c2')

        plt.subplot(nrow, ncol, ncol*2+8)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['iou_f2_c2'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['iou_f2_c2'], 'r--', label='te')
        plt.legend()
        plt.title('iou_f2_c2')

        plt.subplot(nrow, ncol, ncol*3+8)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['iou_f3_c2'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['iou_f3_c2'], 'r--', label='te')
        plt.legend()
        plt.title('iou_f3_c2')

        ######################################################

        plt.tight_layout()
        plt.savefig(Path(self.path).joinpath('log.png'))
        plt.close()

    def plot_seal(self):
        ncol = 4
        nrow = 3
        fig = plt.figure(figsize=(ncol*self.fig_scale, nrow*self.fig_scale))
        plt.subplot(nrow, ncol, ncol*0+1)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['accu'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['accu'], 'r--', label='te')
        plt.legend()
        plt.title('Accu.')

        plt.subplot(nrow, ncol, ncol*1+1)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['auroc_avg'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['auroc_avg'], 'r--', label='te')
        plt.legend()
        plt.title('auroc_avg')

        plt.subplot(nrow, ncol, ncol*2+1)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['kappa'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['kappa'], 'r--', label='te')
        plt.legend()
        plt.title('kappa')

        #################################

        plt.subplot(nrow, ncol, ncol*0+2)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['TY_loss'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['TY_loss'], 'r--', label='te')
        plt.legend()
        plt.title('TY_loss')

        plt.subplot(nrow, ncol, ncol*1+2)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['XT_loss_avg'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['XT_loss_avg'], 'r--', label='te')
        plt.legend()
        plt.title('XT_loss_avg')

        plt.subplot(nrow, ncol, ncol*2+2)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['total_loss'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['total_loss'], 'r--', label='te')
        plt.legend()
        plt.title('total_loss')

        #################################

        plt.subplot(nrow, ncol, ncol*0+3)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['f1_c0'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['f1_c0'], 'r--', label='te')
        plt.legend()
        plt.title('f1_c0')

        plt.subplot(nrow, ncol, ncol*1+3)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['f1_c0'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['f1_c0'], 'r--', label='te')
        plt.legend()
        plt.title('f1_c1')

        plt.subplot(nrow, ncol, ncol*2+3)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['f1_c0'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['f1_c0'], 'r--', label='te')
        plt.legend()
        plt.title('f1_c2')

        #################################

        plt.subplot(nrow, ncol, ncol*0+4)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['auroc_c0'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['auroc_c0'], 'r--', label='te')
        plt.legend()
        plt.title('auroc_c0')

        plt.subplot(nrow, ncol, ncol*1+4)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['auroc_c1'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['auroc_c1'], 'r--', label='te')
        plt.legend()
        plt.title('auroc_c1')

        plt.subplot(nrow, ncol, ncol*2+4)
        plt.plot(self.tr_hist['epoch'], self.tr_hist['auroc_c2'], 'b', label='tr')
        plt.plot(self.te_hist['epoch'], self.te_hist['auroc_c2'], 'r--', label='te')
        plt.legend()
        plt.title('auroc_c2')


        plt.tight_layout()
        plt.savefig(Path(self.path).joinpath('log.png'))
        plt.close()



