import numpy as np
import random
import torch
import torchvision.transforms
import torch.nn.functional as F


def horisontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets

def vertical_flip(images, targets):
	images = torch.flip(images, [-2])
	targets[:, 3] = 1 - targets[:,3]
	return images, targets

def color_jitter(images, targets):
    images = torchvision.transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0)(images)
    return images, targets

def random_cutout(images, targets):
    channels, x_size, y_size = images.shape
    
    n_cutouts = random.randint(1,2)
    for _ in range(n_cutouts):
        ax, ay = random.randint(0, x_size),random.randint(0, y_size)

        cutoff_size_x = random.randint(int(x_size*0.2), int(x_size*0.4))
        cutoff_size_y = random.randint(int(y_size*0.2), int(y_size*0.4))
       
        bx, by = ax + cutoff_size_x, ay + cutoff_size_y
        if bx > x_size:
            bx = x_size
        if by > y_size:
            by = y_size

        cut_out_zeros = torch.zeros(channels, bx-ax, by-ay)     
        images[:, ax:bx, ay:by] = cut_out_zeros

    return images,targets


