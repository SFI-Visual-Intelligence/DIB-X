import torch
import os
import numpy as np

'''
Out[4]: dict_keys(['iter', 'epoch', 'history', 'args', 'model_states', 'optim_states'])
['history'] => ['best', 'tr', 'te']
['args']
'''

class LoadCurrentAcc():
    def __init__(self, main_dir, data, model, epoch, setup=None, name='current_acc.tar'):
        self.main_dir = main_dir
        self.data = data
        self.model = model
        self.setup = setup
        self.epoch = epoch
        self.name = name
        if self.setup == None:
            self.loaded_hist = torch.load(os.path.join(main_dir, data, model, '%d_' % epoch + name),
                                          map_location=torch.device('cpu'))
        else:
            self.loaded_hist = torch.load(os.path.join(main_dir, data, model, setup, '%d_' % epoch + name), map_location=torch.device('cpu'))
        print('\n', '%s_%s_%s_%s' % (data, model, setup, epoch), '\n',
              self.loaded_hist['history']['best'])

#####################################
# import torch
# import os
# from test_repo.load_current_acc import LoadCurrentAcc

image_dir = '/Users/changkyu/Downloads/P3/'
data = 'echo-0.8'
model = 'vibi'
setup = 'b10-k4-n10'
epoch = 610
hist = LoadCurrentAcc(image_dir, data, model, epoch, setup)
print(hist.loaded_hist['history']['best']['confusion']/hist.loaded_hist['history']['best']['confusion'].sum(1))