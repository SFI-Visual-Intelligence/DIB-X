import torch
import numpy as np
import os


dtype = 'echo'  # 'seal' or 'echo'
method = 'dibx' # 'dibx' 'classifieronly'
beta = 'b00'
gamma = 'g00'
default = '/Users/changkyu/Desktop/Paper3/checkpoints'
loca = os.path.join(default, dtype, method, '%s_%s' %(beta, gamma))

echo_b00_g00 = torch.load(os.path.join(loca, 'current_acc.tar'), map_location=torch.device('cpu'))
conf_echo_b00_g00 = echo_b00_g00['history']['best']['confusion']
###################################################################################################

dtype = 'echo'  # 'seal' or 'echo'
method = 'dibx' # 'dibx' 'classifieronly'
beta = 'b005'
gamma = 'g00'
default = '/Users/changkyu/Desktop/Paper3/checkpoints'

if dtype == 'seal':
    loca = os.path.join(default, dtype, method, beta)
elif dtype == 'echo':
    loca = os.path.join(default, dtype, method, '%s_%s' %(beta, gamma))

echo_b005_g00 = torch.load(os.path.join(loca, 'current_acc.tar'), map_location=torch.device('cpu'))
conf_echo_b005_g00 = echo_b005_g00['history']['best']['confusion']
###################################################################################################

dtype = 'echo'  # 'seal' or 'echo'
method = 'dibx' # 'dibx' 'classifieronly'
beta = 'b005'
gamma = 'g01'
default = '/Users/changkyu/Desktop/Paper3/checkpoints'

if dtype == 'seal':
    loca = os.path.join(default, dtype, method, beta)
elif dtype == 'echo':
    loca = os.path.join(default, dtype, method, '%s_%s' %(beta, gamma))

echo_b005_g01 = torch.load(os.path.join(loca, 'current_acc.tar'), map_location=torch.device('cpu'))
conf_echo_b005_g01 = echo_b005_g01['history']['best']['confusion']
###################################################################################################

dtype = 'echo'  # 'seal' or 'echo'
method = 'dibx' # 'dibx' 'classifieronly'
beta = 'b005'
gamma = 'g005'
default = '/Users/changkyu/Desktop/Paper3/checkpoints'

if dtype == 'seal':
    loca = os.path.join(default, dtype, method, beta)
elif dtype == 'echo':
    loca = os.path.join(default, dtype, method, '%s_%s' %(beta, gamma))

echo_b005_g005 = torch.load(os.path.join(loca, 'current_acc.tar'), map_location=torch.device('cpu'))
conf_echo_b005_g005 = echo_b005_g005['history']['best']['confusion']

###################################################################################################

dtype = 'echo'  # 'seal' or 'echo'
method = 'classifieronly' # 'dibx' 'classifieronly'
default = '/Users/changkyu/Desktop/Paper3/checkpoints'
loca = os.path.join(default, dtype, method)

echo_co = torch.load(os.path.join(loca, 'current_acc.tar'), map_location=torch.device('cpu'))
conf_echo_co = echo_co['history']['best']['confusion']
###################################################################################################


def conf_metrics(conf):
    conf_norm = conf/conf.sum(1)
    accu = (conf.diagonal().sum())/conf.sum()
    recall = (conf/conf.sum(1)).diagonal()
    precision = (conf/conf.sum(0)).diagonal()
    f1_class = 2 * precision * recall / (precision + recall)
    f1_macro = f1_class.mean()
    return recall, precision, f1_class, f1_macro, accu, conf_norm

for c in [conf_echo_b00_g00, conf_echo_b005_g00, conf_echo_b005_g005, conf_echo_b005_g01, conf_echo_co]:
    r, p, f1c, f1m, a, conf_norm = conf_metrics(c)
    print(r, p, f1c, f1m, a, '\n', conf_norm)



for i, d in enumerate([echo_b00_g00, echo_b005_g00, echo_b005_g005, echo_b005_g01, echo_co]):
    print(i, '\n', d['history']['best'])








# dtype = 'seal'  # 'seal' or 'echo'
# method = 'dibx' # 'dibx' 'classifieronly'
# beta = 'b00'
# gamma = 'g00'
# default = '/Users/changkyu/Desktop/Paper3/checkpoints'
#
# if dtype == 'seal':
#     loca = os.path.join(default, dtype, method, beta)
# elif dtype == 'echo':
#     loca = os.path.join(default, dtype, method, '%s_%s' %(beta, gamma))
#
# seal_b00 = torch.load(os.path.join(loca, 'current_acc.tar'), map_location=torch.device('cpu'))
# conf_b00 = seal_b00['history']['best']['confusion']
# ###################################################################################################
#
# dtype = 'seal'  # 'seal' or 'echo'
# method = 'dibx' # 'dibx' 'classifieronly'
# beta = 'b01'
# gamma = 'g00'
# default = '/Users/changkyu/Desktop/Paper3/checkpoints'
#
# if dtype == 'seal':
#     loca = os.path.join(default, dtype, method, beta)
# elif dtype == 'echo':
#     loca = os.path.join(default, dtype, method, '%s_%s' %(beta, gamma))
#
# seal_b01 = torch.load(os.path.join(loca, 'current_acc.tar'), map_location=torch.device('cpu'))
# conf_b01 = seal_b01['history']['best']['confusion']
# ###################################################################################################
#
# dtype = 'seal'  # 'seal' or 'echo'
# method = 'classifieronly' # 'dibx' 'classifieronly'
# default = '/Users/changkyu/Desktop/Paper3/checkpoints'
# loca = os.path.join(default, dtype, method)
#
# seal_co = torch.load(os.path.join(loca, 'current_acc.tar'), map_location=torch.device('cpu'))
# conf_co = seal_co['history']['best']['confusion']
# ###################################################################################################
#
# def conf_metrics(conf):
#     conf_norm = conf/conf.sum(1)
#     accu = (conf.diagonal().sum())/conf.sum()
#     recall = (conf/conf.sum(1)).diagonal()
#     precision = (conf/conf.sum(0)).diagonal()
#     f1_class = 2 * precision * recall / (precision + recall)
#     f1_macro = f1_class.mean()
#     return recall, precision, f1_class, f1_macro, accu, conf_norm
#
# for c in [conf_b00, conf_b01, conf_co]:
#     r, p, f1c, f1m, a, conf_norm = conf_metrics(c)
#     print(r, p, f1c, f1m, a, '\n', conf_norm)




'''keys
dict_keys(['iter', 'epoch', 'history', 'args', 'model_states', 'optim_states'])

dict_keys(['epoch', 'TY_loss', 'XT_loss_avg', 
                    'XT_loss_f0', 'XT_loss_f1', 'XT_loss_f2', 'XT_loss_f3', 
                    'MP_loss_avg', 
                    'MP_loss_f0', 'MP_loss_f1', 'MP_loss_f2', 'MP_loss_f3', 
                    'total_loss', 'accu', 'confusion', 
                    'f1_c0', 'f1_c1', 'f1_c2', 
                    'auroc_avg', 
                    'auroc_c0', 'auroc_c1', 'auroc_c2', 
                    'kappa', 
                    'iou_avg_c0', 'iou_avg_c1', 'iou_avg_c2', 
                    'iou_f0_c0', 'iou_f0_c1', 'iou_f0_c2', 
                    'iou_f1_c0', 'iou_f1_c1', 'iou_f1_c2', 
                    'iou_f2_c0', 'iou_f2_c1', 'iou_f2_c2', 
                    'iou_f3_c0', 'iou_f3_c1', 'iou_f3_c2'])

####################################################
SEAL
####################################################
b00
####################################################
{'epoch': 689,
 'TY_loss': 0.5117196187178293,
 'XT_loss_avg': 0.0,
 'XT_loss_f0': 0,
 'XT_loss_f1': 0,
 'XT_loss_f2': 0,
 'XT_loss_f3': 0,
 'MP_loss_avg': 0.0,
 'MP_loss_f0': 0,
 'MP_loss_f1': 0,
 'MP_loss_f2': 0,
 'MP_loss_f3': 0,
 'total_loss': 0.5117196187178293,
 'accu': 0.93999999888738,
 'confusion': tensor([[41.1200,  0.5120,  0.1707],
         [ 0.0000, 40.2907,  1.4880],
         [ 1.4880,  3.8773, 36.4133]]),
 'f1_c0': 0.9743609081904093,
 'f1_c1': 0.9322988543510436,
 'f1_c2': 0.9118391006787618,
 'auroc_avg': 0.9831081380844117,
 'auroc_c0': 0.9955045250256856,
 'auroc_c1': 0.9844014878273011,
 'auroc_c2': 0.9694183039665222,
 'kappa': 0.9100102167129517,
 'iou_avg_c0': 0,
 'iou_avg_c1': 0,
 'iou_avg_c2': 0,
 'iou_f0_c0': 0,
 'iou_f0_c1': 0,
 'iou_f0_c2': 0,
####################################################
b01
####################################################
{'epoch': 1214,
 'TY_loss': 0.5884603774547577,
 'XT_loss_avg': 0.025518230269352595,
 'XT_loss_f0': 0,
 'XT_loss_f1': 0,
 'XT_loss_f2': 0,
 'XT_loss_f3': 0,
 'MP_loss_avg': 0.0,
 'MP_loss_f0': 0,
 'MP_loss_f1': 0,
 'MP_loss_f2': 0,
 'MP_loss_f3': 0,
 'total_loss': 0.6139786008199056,
 'accu': 0.9213333296775817,
 'confusion': tensor([[40.7787,  0.8533,  0.1707],
         [ 0.6827, 40.5840,  0.5120],
         [ 2.6587,  5.0960, 34.0240]]),
 'f1': 0.9213333296775817,
 'auroc_avg': 0.979256772518158,
 'auroc_c0': 0.9927529160181682,
 'auroc_c1': 0.9799070936838786,
 'auroc_c2': 0.9651102889378865,
 'kappa': 0.8820307375590006,
 'iou_avg_c0': 0,
 'iou_avg_c1': 0,
 'iou_avg_c2': 0,
 'iou_f0_c0': 0,
 'iou_f0_c1': 0,
 'iou_f0_c2': 0,
 'iou_f1_c0': 0,
 'iou_f1_c1': 0,
 'iou_f1_c2': 0,
 'iou_f2_c0': 0,
 'iou_f2_c1': 0,
 'iou_f2_c2': 0,
 'iou_f3_c0': 0,
 'iou_f3_c1': 0,
 'iou_f3_c2': 0}
####################################################
classifieronly
####################################################
{'epoch': 525,
 'TY_loss': 0.5120896972020467,
 'XT_loss_avg': 0.0,
 'XT_loss_f0': 0,
 'XT_loss_f1': 0,
 'XT_loss_f2': 0,
 'XT_loss_f3': 0,
 'MP_loss_avg': 0.0,
 'MP_loss_f0': 0,
 'MP_loss_f1': 0,
 'MP_loss_f2': 0,
 'MP_loss_f3': 0,
 'total_loss': 0.5120896972020467,
 'accu': 0.864000002861023,
 'confusion': tensor([[40.6320,  1.0000,  0.1707],
         [ 0.3413, 41.2667,  0.1707],
         [ 2.8053, 12.5360, 26.4373]]),
 'f1_c0': 0.9497468810081482,sst
 'f1_c1': 0.8550486868222554,
 'f1_c2': 0.7688624575932821,
 'auroc_avg': 0.9758497314453125,
 'auroc_c0': 0.9893910516103108,
 'auroc_c1': 0.9799625992774963,
 'auroc_c2': 0.9581954838434855,
 'kappa': 0.7960340030988058,
 'iou_avg_c0': 0,
 'iou_avg_c1': 0,
 'iou_avg_c2': 0,
 'iou_f0_c0': 0,
'''
