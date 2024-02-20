import numpy as np
import torch
import os
import sys


current_dir = os.path.dirname(os.path.abspath(__file__))
# current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, '..', '..', '..'))
sys.path.append(current_dir)

import tempfile
os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()

from solver import Solver
from args_collection import parse_args

def main():
    args = parse_args()

    # if args.gpu == 'springfield':
    import wandb
    wandb.init(project="%s_%s_%s" % (args.data, args.model, args.gpu), entity = "changkyuchoi")


    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    ## print-option
    np.set_printoptions(precision=4) # upto 4th digits for floating point output
    torch.set_printoptions(precision=4)
    print('\n[ARGUMENTS]\n', args)

    ## cuda
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda True")
    args.cuda = (args.cuda and torch.cuda.is_available())
    
    exp = Solver(data_type='test', args=args) # data_type='test', 'valid', 'mixed'
    if args.pretrain_epoch > 0:
        exp.pretrain(tr_data_type='train') # pretrain classiier (decode_cnn)
    exp.train(tr_data_type='train')

if __name__ == "__main__":
    main()

