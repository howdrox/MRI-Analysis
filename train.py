import json
import os
import os.path as osp
from tqdm import tqdm
from dataclasses import dataclass

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from datetime import datetime

from dataset import BrainSegmentationDataset as Dataset
from logger import Logger
from loss import DiceLoss  # or your BCEDiceLoss if you replaced it
from transform import transforms
from unet import UNet
from utils import log_images, dsc

def worker_init(worker_id):
    np.random.seed(42 + worker_id)

def data_loaders(args):
    dataset_train, dataset_valid = datasets(args)

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        pin_memory=False,  # no GPU
        worker_init_fn=worker_init,  # Use the standalone function
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=False,  # no GPU
        worker_init_fn=worker_init,  # Use the standalone function
    )

    return loader_train, loader_valid


def datasets(args):
    train = Dataset(
        images_dir=args.images,
        subset="train",
        image_size=args.image_size,
        transform=transforms(scale=args.aug_scale, angle=args.aug_angle, flip_prob=0.5),
    )
    valid = Dataset(
        images_dir=args.images,
        subset="validation",
        image_size=args.image_size,
        random_sampling=False,
    )
    return train, valid


def dsc_per_volume(validation_pred, validation_true, patient_slice_index):
    dsc_list = []
    num_slices = np.bincount([p[0] for p in patient_slice_index])
    index = 0
    for p in range(len(num_slices)):
        y_pred = np.array(validation_pred[index : index + num_slices[p]])
        y_true = np.array(validation_true[index : index + num_slices[p]])
        dsc_list.append(dsc(y_pred, y_true))
        index += num_slices[p]
    return dsc_list


def log_loss_summary(logger, loss, step, prefix=""):
    logger.scalar_summary(prefix + "loss", np.mean(loss), step)


def makedirs(args):
    os.makedirs(args.weights, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)


def snapshotargs(args):
    args_file = os.path.join(args.logs, "args.json")
    with open(args_file, "w") as fp:
        json.dump(vars(args), fp)

@dataclass
class Args:
    device = 'cuda:0'
    batch_size = 16
    epochs = 5
    lr = 0.0003
    workers = 0
    vis_images = 200
    vis_freq = 10
    weights = './weights'
    logs = './logs'
    images = 'BrainMRI/kaggle_3m'
    image_size = 256
    aug_scale = 0.05
    aug_angle = 15

def main():
    args = Args()

    assert osp.exists(args.images), "Please download the dataset and set the correct path" 

    # save configs
    makedirs(args)
    snapshotargs(args)

    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)

    # build dataset
    loader_train, loader_valid = data_loaders(args)
    loaders = {"train": loader_train, "valid": loader_valid}

    # build model
    unet = UNet(in_channels=Dataset.in_channels, out_channels=Dataset.out_channels)
    unet.to(device)

    checkpoint_path = os.path.join(args.weights, "unet.pt")
    if os.path.exists(checkpoint_path):
        unet.load_state_dict(torch.load(checkpoint_path))
        print("Loaded checkpoint from", checkpoint_path)

    # build optimizer
    optimizer = optim.Adam(unet.parameters(), lr=args.lr)

    # build metric
    dsc_loss = DiceLoss()

    # build loggers (use tensorboard to visualize the loss curves)
    best_validation_dsc = 0.0

    logger = Logger(args.logs)
    loss_train = []
    loss_valid = []

    step = 0

    for epoch in range(args.epochs):
        for phase in ["train", "valid"]:
            print("epoch {}, phase {}, total step {}".format(epoch, phase, step))

            if phase == "train":
                unet.train()
            else:
                unet.eval()

            validation_pred = []
            validation_true = []

            for i, data in enumerate(loaders[phase]):
                if phase == "train":
                    step += 1

                x, y_true = data
                x, y_true = x.to(device), y_true.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_pred = unet(x)

                    loss = dsc_loss(y_pred, y_true)

                    if phase == "valid":
                        loss_valid.append(loss.item())
                        y_pred_np = y_pred.detach().cpu().numpy()
                        validation_pred.extend(
                            [y_pred_np[s] for s in range(y_pred_np.shape[0])]
                        )
                        y_true_np = y_true.detach().cpu().numpy()
                        validation_true.extend(
                            [y_true_np[s] for s in range(y_true_np.shape[0])]
                        )
                        if (epoch % args.vis_freq == 0) or (epoch == args.epochs - 1):
                            if i * args.batch_size < args.vis_images:
                                tag = "image/{}".format(i)
                                num_images = args.vis_images - i * args.batch_size
                                logger.image_list_summary(
                                    tag,
                                    log_images(x, y_true, y_pred)[:num_images],
                                    step,
                                )

                    if phase == "train":
                        loss_train.append(loss.item())
                        loss.backward()
                        optimizer.step()

                if phase == "train" and (step + 1) % 10 == 0:
                    log_loss_summary(logger, loss_train, step)
                    print("epoch {}, phase {}, step {}, loss_train, {:.3f}".format(
                            epoch, phase, step, np.mean(loss_train)))
                    loss_train = []

            if phase == "valid":
                log_loss_summary(logger, loss_valid, step, prefix="val_")
                print("epoch {} | val_loss: {}".format(epoch + 1, np.mean(loss_valid)))
                mean_dsc = np.mean(
                    dsc_per_volume(
                        validation_pred,
                        validation_true,
                        loader_valid.dataset.patient_slice_index,
                    )
                )
                logger.scalar_summary("val_dsc", mean_dsc, step)
                print("epoch {} | val_dsc: {}".format(epoch+1, mean_dsc))
                if mean_dsc > best_validation_dsc:
                    best_validation_dsc = mean_dsc
                    torch.save(unet.state_dict(), os.path.join(args.weights, "unet.pt"))
                loss_valid = []

    print("Best validation mean DSC: {:4f}".format(best_validation_dsc))

if __name__ == "__main__":
    start_time = datetime.now()
    print("Start time:", start_time)
    main()
    print("End time:", datetime.now())
    print("Total time:", datetime.now() - start_time)