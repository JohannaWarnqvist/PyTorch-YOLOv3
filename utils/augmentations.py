import torch
import torchvision.transforms
import torch.nn.functional as F
import numpy as np


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets

def color_jitter(images, targets):
    images = ColorJitter(images,brightness=0, contrast=0, saturation=0, hue=0)
    return images, targets
