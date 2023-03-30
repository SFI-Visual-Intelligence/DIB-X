import argparse
import os
import sys
from utils.general_utils import str2bool

def parse_args():
    parser = argparse.ArgumentParser(description = 'DIB-X')
    current_dir = os.getcwd()
    parser.add_argument('--gpu', default='sp', type=str, help='springfield or colab')
    parser.add_argument('--model', default='dibx', type=str, help='model choice: dibx, vibi, gradcam, lime')
    ############################################################################################
    parser.add_argument('--data', default='echograms', type=str, help='dataset name: echograms, seal')
    parser.add_argument('--explainer_in_depth', default=4, type = int, help='explainer input depth')
    parser.add_argument('--explainer_out_depth', default=4, type = int, help='explainer output depth')
    parser.add_argument('--nmb_category', default=3, type = int, help='number of classes')
    ############################################################################################
    parser.add_argument('--pretrain_epoch', default=200, type = int, help='pretrain the classifier before end-to-end training')
    parser.add_argument('--pretrain_accu', default=0.5, type = float, help='pretraining stopping criteria wrt tr_accu')
    ############################################################################################
    parser.add_argument('--dibx_beta', default=0.02, type = float,
                        help = 'beta balancing between IXT and ITY')
    parser.add_argument('--dibx_gamma', default=0.00, type = float,
                        help = 'gamma regulating KLDloss btw learned mask and and its prior')
    parser.add_argument('--dibx_kernel_size', default=9, type = int, help = 'kernel size for DIB:entropy')
    parser.add_argument('--dibx_MI_patch_split', type=int, default=4, help='the patch split count for MI(T,X), if 4 , patch_size:32')
    ############################################################################################
    # parser.add_argument('--vibi_beta', default=0.01, type=float,
    #                     help='beta balancing between IXT and ITY')
    # parser.add_argument('--vibi_gamma', default=0, type=float,
    #                     help='gamma regulating KLDloss btw learned mask and and its prior')
    # parser.add_argument('--vibi_tau', default=0.5, type=float, help='temperature for RelaxedOneHotCategorical')
    # parser.add_argument('--vibi_num_avg_sample', default=4, type=int, help='vibi for continous mask')
    # parser.add_argument('--vibi_K', default= 4, type=int, help='vibi K samples for each mask')
    ############################################################################################
    parser.add_argument('--save_accuracy', default=0.6, type = float, help = 'minumum prediction accuracy for save_checkpoint_function')
    parser.add_argument('--bn', default=True, type=str2bool, help='batch-norm-layer')
    parser.add_argument('--chunk_size', default=8, type = int, help = 'superpixel dim for explainability')
    parser.add_argument('--global_epoch', default=-1, type = int, help = 'global epoch count')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
    parser.add_argument('--lr_exp', default=3e-4, type = float, help = 'learning rate')
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--default_dir', default=current_dir, type = str, help='default directory path')
    parser.add_argument('--checkpoint_dir', default='checkpoints', type = str, help='checkpoint directory path')
    parser.add_argument('--epoch', default=1001, type = int, help = 'epoch number')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--window_dim', type=int, default=128, help='window size')
    parser.add_argument('--saliency_visual_threshold', default=0.9, type=float, help='ExplainVisual_saliency criteria (max=1)')
    return parser.parse_args([])

