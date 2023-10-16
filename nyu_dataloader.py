from random import choice
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import random
import numpy as np
from PIL import Image

def rgb2ycbcr(rgb_image):
    """convert rgb into ycbcr"""

    if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
        raise ValueError("input image is not a rgb image")
    rgb_image = rgb_image.astype(np.float32)
    transform_matrix = np.array([[0.257, 0.564, 0.098],
                                 [-0.148, -0.291, 0.439],
                                 [0.439, -0.368, -0.071]])
    shift_matrix = np.array([16, 128, 128])
    ycbcr_image = np.zeros(shape=rgb_image.shape)
    w, h, _ = rgb_image.shape
    for i in range(w):
        for j in range(h):
            ycbcr_image[i, j, :] = np.dot(transform_matrix, rgb_image[i, j, :]) + shift_matrix
    return ycbcr_image

class NYU_v2_dataset(Dataset):
    """NYUDataset."""

    def __init__(self, root_dir, scale=8, train=True, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.scale = scale

        if train:
            self.depths = np.load('%s/train_depth_split.npy' % root_dir)
            self.images = np.load('%s/train_images_split.npy' % root_dir)
        else:
            self.depths = np.load('%s/test_depth.npy' % root_dir)
            self.images = np.load('%s/test_images_v2.npy' % root_dir)
        self.train = train

    def __len__(self):
        return self.depths.shape[0]

    @staticmethod
    def random_crop_params(img, output_size):
        h, w = img.shape[:2]
        out_h, out_w = output_size
        if w == out_w and h == out_h:
            return 0, 0, h, w
        i = random.randint(0, h - out_h)
        j = random.randint(0, w - out_w)
        return i, j, out_h, out_w

    def __getitem__(self, idx):
        depth = self.depths[idx]
        image = self.images[idx]

        s = self.scale
        if self.train:
            i, j, out_h, out_w = self.random_crop_params(image, (96, 96))
            image = image[i: i + out_h, j: j + out_w, :]
            depth = depth[i: i + out_h, j: j + out_w]

        h, w = depth.shape
        target = np.array(Image.fromarray(depth).resize((w // s, h // s), Image.BICUBIC))
        image = (rgb2ycbcr(image*255))[:, :, [0]]
        image = (image - image.min()) / (image.max() - image.min() + 0.001)

        if self.transform:
            image = self.transform(image).float()
            depth = self.transform(np.expand_dims(depth, 2)).float()
            target = self.transform(np.expand_dims(target, 2)).float()

        sample = {'guidance': image, 'target': target, 'gt': depth}

        return sample

