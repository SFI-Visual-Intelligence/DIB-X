import numpy as np

# from utils.np import getGrid, linear_interpolation, nearest_interpolation

class EchosounderData():

    def __init__(self, data_in, cls_labels_in, seg_labels_in,
                 mask_prior_in=None,
                 augmentation_function=None,
                 label_transform_function=None,
                 data_transform_function=None):
        """
        A dataset is used to draw random samples
        :param samplers: The samplers used to draw samples
        :param window_size: expected window size
        :param n_samples:
        :param frequencies:
        :param sampler_probs:
        :param augmentation_function:
        :param label_transform_function:
        :param data_transform_function:
        """
        self.data_in = data_in # max 2 (seabed feature)
        self.cls_labels_in = cls_labels_in
        self.seg_labels_in = seg_labels_in
        self.mask_prior_in = mask_prior_in
        self.n_samples = len(cls_labels_in)
        # self.augmentation_function = augmentation_function
        # self.label_transform_function = label_transform_function
        # self.data_transform_function = data_transform_function

    def __getitem__(self, index):

        data = self.data_in[index]
        cls_labels = self.cls_labels_in[index]
        seg_labels = self.seg_labels_in[index]

        # # Apply augmentation
        # if self.augmentation_function is not None:
        #     data, cls_labels = self.augmentation_function(data, cls_labels)
        #
        # # Apply label-transform-function
        # if self.label_transform_function is not None:
        #     data, cls_labels = self.label_transform_function(data, cls_labels)
        #
        # # Apply data-transform-function
        # if self.data_transform_function is not None:
        #     data = self.data_transform_function(data)
        #     data = np.squeeze(data)
        #
        # if not np.isscalar(cls_labels):
        #     cls_labels = cls_labels.astype('int16')

        if self.mask_prior_in is None:
            return data, cls_labels, seg_labels, None
        else:
            mask_prior = self.mask_prior_in[index]
            return data, cls_labels, seg_labels, mask_prior


    def __len__(self):
        return self.n_samples

class SealData():

    def __init__(self, data_in, cls_labels_in,
                 ):
        """
        A dataset is used to draw random samples
        :param samplers: The samplers used to draw samples
        :param window_size: expected window size
        :param n_samples:
        :param frequencies:
        :param sampler_probs:
        """
        self.data_in = data_in
        self.cls_labels_in = cls_labels_in
        self.n_samples = len(cls_labels_in)

    def __getitem__(self, index):

        data = self.data_in[index]
        cls_labels = self.cls_labels_in[index]

        if not np.isscalar(cls_labels):
            cls_labels = cls_labels.astype('int16')

        return data, cls_labels

    def __len__(self):
        return self.n_samples


#
#
# def patch_splitter(sample, after_size=32, before_size=256, step=32, half_idx=0):
#     slice_indices = np.arange(after_size, before_size, step=step)
#     patches = np.asarray(np.split(sample, slice_indices, axis=-1))
#     if len(np.shape(patches)) == 4: # echosounder patch
#         patches = np.reshape(np.asarray(np.split(patches, slice_indices, axis=-2)), (-1, 4, 32, 32))
#     elif len(np.shape(patches)) == 3: # label patch
#         patches = np.reshape(np.asarray(np.split(patches, slice_indices, axis=-2)), (-1, 32, 32))
#
#     if half_idx == 0:
#         patches = patches [:32]
#     else:
#         patches = patches[32:]
#     return patches
#
# def label_scalar(l_patches, criteria=16):
#     l_scalars = []
#     for l_patch in l_patches:
#         unique = np.unique(l_patch)
#         if np.isin(-1, unique):
#             l_scalar = -1
#         else:
#             l_vec = np.bincount(l_patch.astype(int).reshape(-1), minlength=3)
#             if (l_vec[1] > criteria) or (l_vec[2] > criteria):
#                 if l_vec[1] > l_vec[2]:
#                     l_scalar = 1
#                 else:
#                     l_scalar = 2
#             else:
#                 l_scalar = 0
#         l_scalars.append(l_scalar)
#
#     return np.asarray(l_scalars)
#
# class Dataset():
#
#     def __init__(self, samplers, window_size, frequencies,
#                  n_samples = 1000,
#                  sampler_probs=None,
#                  augmentation_function=None,
#                  label_transform_function=None,
#                  data_transform_function=None):
#         """
#         A dataset is used to draw random samples
#         :param samplers: The samplers used to draw samples
#         :param window_size: expected window size
#         :param n_samples:
#         :param frequencies:
#         :param sampler_probs:
#         :param augmentation_function:
#         :param label_transform_function:
#         :param data_transform_function:
#         """
#
#         self.samplers = samplers
#         self.window_size = window_size
#         self.n_samples = n_samples
#         self.frequencies = frequencies
#         self.sampler_probs = sampler_probs
#         self.augmentation_function = augmentation_function
#         self.label_transform_function = label_transform_function
#         self.data_transform_function = data_transform_function
#
#         # Normalize sampling probabillities
#         if self.sampler_probs is None:
#             self.sampler_probs = np.ones(len(samplers))
#         self.sampler_probs = np.array(self.sampler_probs)
#         self.sampler_probs = np.cumsum(self.sampler_probs).astype(float)
#         self.sampler_probs /= np.max(self.sampler_probs)
#
#     def __getitem__(self, index):
#         #Select which sampler to use
#         i = np.random.rand()
#         sample_idx = np.where(i < self.sampler_probs)[0][0]
#         sampler = self.samplers[sample_idx]
#
#         #Draw coordinate and echogram with sampler
#         center_location, echogram = sampler.get_sample()
#
#         #Get data/labels-patches
#         data, labels = get_crop(echogram, center_location, self.window_size, self.frequencies)
#
#         # Apply augmentation
#         if self.augmentation_function is not None:
#             data, labels, echogram = self.augmentation_function(data, labels, echogram)
#
#         # Apply label-transform-function
#         if self.label_transform_function is not None:
#             data, labels, echogram = self.label_transform_function(data, labels, echogram)
#
#         # Apply data-transform-function
#         if self.data_transform_function is not None:
#             data, labels, echogram, frequencies = self.data_transform_function(data, labels, echogram, self.frequencies)
#
#         labels = labels.astype('int16')
#         return data, sample_idx
#
#     def __len__(self):
#         return self.n_samples
#
#
# class DatasetImg():
#     def __init__(self, samplers,
#                  augmentation_function,
#                  data_transform_function):
#         """
#         A dataset is used to draw random samples
#         :param samplers: The samplers used to draw samples
#         :param window_size: expected window size
#         :param n_samples:
#         :param frequencies:
#         :param sampler_probs:
#         :param augmentation_function:
#         :param label_transform_function:
#         :param data_transform_function:
#         """
#         self.samplers = samplers
#         self.n_samples = int(len(self.samplers) * len(self.samplers[0]))
#         self.augmentation_function = augmentation_function
#         self.data_transform_function = data_transform_function
#
#
#     def __getitem__(self, index):
#         #Select which sampler to use
#         sample_idx = index % len(self.samplers)
#         img_idx = index // len(self.samplers)
#         data = self.samplers[sample_idx][img_idx]
#         # Apply augmentation
#         if self.augmentation_function is not None:
#             data = self.augmentation_function(data)
#         # Apply data-transform-function
#         if self.data_transform_function is not None:
#             data = self.data_transform_function(data)
#         if len(data) == 1:
#             data = np.squeeze(data, 0)
#         return data, sample_idx
#
#     def __len__(self):
#         return self.n_samples
#
#
# def get_crop(echogram, center_location, window_size, freqs):
#     """
#     Returns a crop of data around the pixels specified in the center_location.
#
#     """
#     # Get grid sampled around center_location
#     grid = getGrid(window_size) + np.expand_dims(np.expand_dims(center_location, 1), 1)
#
#     channels = []
#     for f in freqs:
#
#         # Interpolate data onto grid
#         memmap = echogram.data_memmaps(f)[0]
#         data = linear_interpolation(memmap, grid, boundary_val=0, out_shape=window_size)
#         del memmap
#
#         # Set non-finite values (nan, positive inf, negative inf) to zero
#         if np.any(np.invert(np.isfinite(data))):
#             data[np.invert(np.isfinite(data))] = 0
#
#         channels.append(np.expand_dims(data, 0))
#     channels = np.concatenate(channels, 0)
#
#     # labels = nearest_interpolation(echogram.label_memmap(), grid, boundary_val=-100, out_shape=window_size)
#     labels = nearest_interpolation(echogram.label_memmap(), grid, boundary_val=-100, out_shape=window_size)
#
#     return channels, labels

