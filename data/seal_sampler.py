import os
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data

import paths
from data.dataset import SealData


class SampleSealData128():
    '''
    TR_data 3000 samples (1000 each class)
    TE_data 750 samples
    Val_data 150 samples
    '''

    def __init__(self, args):
        self.args = args
        self.path_to_echograms = paths.path_to_echograms()

    def for_train(self):
        data_tr = torch.load(os.path.join(self.path_to_echograms, 'seal_data_tr_3class.pt'))
        cls_label_tr = torch.load(os.path.join(self.path_to_echograms, 'seal_label_tr_3class.pt'))  # class
        dataset_train = SealData(
            data_tr,
            cls_label_tr)
        return dataset_train

    def for_test(self):
        data_te = torch.load(os.path.join(self.path_to_echograms, 'seal_data_te_3class.pt'))
        cls_label_te = torch.load(os.path.join(self.path_to_echograms, 'seal_label_te_3class.pt'))  # class
        dataset_test = SealData(
            data_te,
            cls_label_te)
        return dataset_test

    def for_val(self):
        data_val = torch.load(os.path.join(self.path_to_echograms, 'seal_data_val_3class.pt'))
        cls_label_val = torch.load(os.path.join(self.path_to_echograms, 'seal_label_val_3class.pt'))  # class
        dataset_val = SealData(
            data_val,
            cls_label_val)
        return dataset_val



# def seal_data_generator(sub = 'bg', sample_size=1300):
#     from PIL import Image as PImage
#     dir = '/Users/changkyu/Desktop'
#     sub_dir = os.path.join(dir, sub)
#     imagesList = os.listdir(sub_dir)
#
#     loadedImages = []
#     for image in imagesList:
#         if image[:6] in ['westic', 'canada']:
#             img = PImage.open(os.path.join(sub_dir, image))
#             loadedImages.append(np.asarray(img)/255.0)
#
#     sample_idx = sorted(np.random.choice(len(loadedImages), size=sample_size, replace=False))
#     npld = np.asarray([loadedImages[i] for i in sample_idx])
#     torch.save(npld.transpose(0, 3, 1, 2), os.path.join(dir, 'seal_%s.pt') % sub)
#     torch.save(sample_idx, os.path.join(dir, 'seal_%s_idx.pt') % sub)
#     return None
#
# def seal_training_test():
#     dir = '/Users/changkyu/Desktop'
#     bg = torch.load(os.path.join(dir, 'seal_bg.pt'))
#     harp = torch.load(os.path.join(dir, 'seal_harp.pt'))
#     hood = torch.load(os.path.join(dir, 'seal_hood.pt'))
#
#     idx = np.arange(len(bg))
#     np.random.shuffle(idx)
#     tr_idx = idx[:1000]
#     te_idx = idx[1000:1250]
#     val_idx = idx[1250:]
#
#     '''
#     BG: class 0
#     HARP: class 1
#     HOOD: class 2
#     '''
#     tr_data = []
#     te_data = []
#     val_data = []
#     tr_label = []
#     te_label = []
#     val_label = []
#     for i in range(len(idx)):
#         if i in tr_idx:
#             tr_data.append(bg[i])
#             tr_label.append(0)
#             tr_data.append(harp[i])
#             tr_label.append(1)
#             tr_data.append(hood[i])
#             tr_label.append(2)
#         elif i in te_idx:
#             te_data.append(bg[i])
#             te_label.append(0)
#             te_data.append(harp[i])
#             te_label.append(1)
#             te_data.append(hood[i])
#             te_label.append(2)
#         elif i in val_idx:
#             val_data.append(bg[i])
#             val_label.append(0)
#             val_data.append(harp[i])
#             val_label.append(1)
#             val_data.append(hood[i])
#             val_label.append(2)
#
#     torch.save(tr_data, os.path.join(dir, 'seal_data_tr_3class.pt'))
#     torch.save(te_data, os.path.join(dir, 'seal_data_te_3class.pt'))
#     torch.save(val_data, os.path.join(dir, 'seal_data_val_3class.pt'))
#     torch.save(tr_label, os.path.join(dir, 'seal_label_tr_3class.pt'))
#     torch.save(te_label, os.path.join(dir, 'seal_label_te_3class.pt'))
#     torch.save(val_label, os.path.join(dir, 'seal_label_val_3class.pt'))
#
#     return None

