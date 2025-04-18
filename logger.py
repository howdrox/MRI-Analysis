import numpy as np
import torch
import torchvision 

from torch.utils.tensorboard import SummaryWriter


class Logger(object):

    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
        self.writer.flush()

    def image_summary(self, tag, image, step):
        self.writer.add_image(tag, image, step)
        self.writer.flush()

    def image_list_summary(self, tag, images, step):
        if len(images) == 0:
            return
        
        images = np.array(images)
        images = torch.tensor(images).permute(0, 3, 1, 2)   # N,H,W,C -> N,C,H,W

        grid = torchvision.utils.make_grid(images)
        self.writer.add_image(tag, grid, step)
        self.writer.flush()
