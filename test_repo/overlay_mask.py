import torch
import os
from matplotlib.colors import ListedColormap
import torch.nn.functional as F
import matplotlib.pyplot as plt
import copy
from pathlib import Path
import sys
current_dir = os.getcwd()
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, '..'))
from utils.visualize import ExplainVisual


class PostAnalysisVisual(ExplainVisual):
    def __init__(self, data_setup, **kwargs):
        self.figsize_scale = 2
        self.epoch = kwargs['epoch']
        self.idx = kwargs['idx']
        self.data_setup = data_setup
        self.filename = '%d_%d_logit_n_mask_visual.pt' % (self.epoch, self.idx)
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.load = torch.load(os.path.join(kwargs['image_dir'], self.data_setup, self.filename), map_location=self.device)
        self.data = self.load['data']  # seal or echo
        self.threshold = kwargs['threshold']
        self.image_dir = os.path.join(kwargs['image_dir'], self.data_setup, str(self.threshold))
        if not os.path.isdir(self.image_dir):
            os.mkdir(self.image_dir)
        self.batch = self.load['x_raw']
        self.label = self.load['y_raw']
        self.label_approx = self.load['logit'].argmax(-1)
        self.encoded_mask = self.load['mask']
        for i in range(len(self.encoded_mask)):
            if (self.encoded_mask[i].max() > 0) and (self.encoded_mask[i].max() < 1):
                self.encoded_mask[i] = self.encoded_mask[i]/self.encoded_mask[i].max()

        self.chunk_size = 8
        if self.data == 'mnist':
            self.chunk_size = 4

        if self.encoded_mask.shape[-1] == self.batch.shape[-1]:
            self.encoded_mask = F.max_pool2d(self.encoded_mask, self.chunk_size, self.chunk_size)

        self.seg_label = self.load['y_seg']
        self.prior = self.load['m_prior']
        self.cmap = 'gist_heat_r'
        self.colormap_v_max_scale = 1.5

        self.rgb_colormap = ListedColormap([[1, 1, 1], [1, 1, 1], [1, 1, 1],
                                            [0, 0, 1], [0, 0, 1], [0, 0, 1],
                                            [1, 0, 0], [1, 0, 0], [1, 0, 0]])
        self.width = self.batch.size(-1)
        self.height = self.batch.size(-2)
        super(PostAnalysisVisual, self).__init__(self.image_dir, self.idx, self.data, self.epoch, self.batch,
                                                 self.encoded_mask, self.label, self.label_approx, self.seg_label, self.prior, self.threshold)
        if self.data == 'echograms':
            self.output = self.visualize_echo()
        else:
            self.output = self.visualize_others()


# import torch
# import os
# from matplotlib.colors import ListedColormap
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import copy
# from pathlib import Path
# import sys
# current_dir = os.getcwd()
# sys.path.append(current_dir)
# sys.path.append(os.path.join(current_dir, '..'))
# from utils.visualize import ExplainVisual
# from test_repo.overlay_mask import PostAnalysisVisual


image_dir = '/Users/changkyu/Downloads/P3/'
data = 'echo-0.8'
model = 'vibi'
setup = 'b10-k4-n10'
data_setup = os.path.join(data, model, setup)
epoch = 610
for idx in [0, 1, 2, 3]:
    for thres in [0.1, 0.8]:
        p = PostAnalysisVisual(epoch=epoch, idx=idx, data_setup=data_setup, threshold=thres, image_dir=image_dir)

