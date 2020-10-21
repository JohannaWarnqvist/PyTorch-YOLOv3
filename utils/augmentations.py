import torch
import torchvision.transforms
import torch.nn.functional as F
import numpy as np


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets

def vertical_flip(images, targets):
	images = torch.flip(images, [-2])
	targets[:, 3] = 1 - targets[:,3]
	return images, targets

def color_jitter(images, targets):
    images = ColorJitter(images,brightness=1, contrast=1, saturation=1, hue=0.5)
    return images, targets
