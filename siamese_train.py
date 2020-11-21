import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from datetime import datetime


from siamese_dataloader import *
from siamese_net import *
import os

import nonechucks as nc
from scipy.stats import multivariate_normal

"""
Get training data
"""


class Config:
    training_dir = "/content/my_track_dataset_train"
    testing_dir = "/content/my_track_dataset_test"
    train_batch_size = 256
    train_number_epochs = 30
    model_save_path = "/content/gdrive/MyDrive/deep_sort/models"
    checkpoint_save_path = "/content/gdrive/MyDrive/deep_sort/models"
    checkpoint_load_path = ""


folder_dataset = dset.ImageFolder(root=Config.training_dir)

transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((128, 128)),
    torchvision.transforms.ToTensor()
])


def get_gaussian_mask():
    # 128 is image size
    x, y = np.mgrid[0:1.0:128j, 0:1.0:128j]  # 128 is input size.
    xy = np.column_stack([x.flat, y.flat])
    mu = np.array([0.5, 0.5])
    sigma = np.array([0.22, 0.22])
    covariance = np.diag(sigma ** 2)
    z = multivariate_normal.pdf(xy, mean=mu, cov=covariance)
    z = z.reshape(x.shape)

    z = z / z.max()
    z = z.astype(np.float32)

    mask = torch.from_numpy(z)

    return mask

siamese_dataset = SiameseTriplet(imageFolderDataset=folder_dataset, transform=transforms, should_invert=False)
net = SiameseNetwork().cuda()

criterion = TripletLoss(margin=1)
optimizer = optim.Adam(net.parameters(), lr=0.0005)  # changed from 0.0005

print(torch.cuda.is_available())

counter = []
loss_history = []
iteration_number = 0

train_dataloader = DataLoader(siamese_dataset, shuffle=False, num_workers=14, batch_size=Config.train_batch_size)

start_iter = 0
start_epoch = 0

if Config.checkpoint_load_path:
    checkpoint = torch.load(Config.checkpoint_load_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_iter = checkpoint['iter']
    start_epoch = checkpoint['epoch']
    iteration_number= checkpoint['ended_iter']
    #loss = checkpoint['loss']
    print("Successfully loaded checkpoint {}".format(Config.checkpoint_load_path))


# Multiply each image with mask to give attention to center of the image.
gaussian_mask = get_gaussian_mask().cuda()

training_start_time = datetime.now()


for epoch in range(start_epoch, Config.train_number_epochs):

    epoch_start_time = datetime.now()

    for i, data in enumerate(train_dataloader, start_iter):

        iteration_start_time = datetime.now()

        iteration_number += 1
        anchor, positive, negative = data
        anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()

        anchor, positive, negative = anchor * gaussian_mask, positive * gaussian_mask, negative * gaussian_mask

        optimizer.zero_grad()

        anchor_out, positive_out, negative_out = net(anchor, positive, negative)

        triplet_loss = criterion(anchor_out, positive_out, negative_out)
        triplet_loss.backward()
        optimizer.step()

        iteration_elapsed_time = datetime.now() - iteration_start_time
        print("Epoch number: {}, iteration: {}, training loss: {}, iteration elapsed time: {}".format(epoch, iteration_number, triplet_loss.item(), iteration_elapsed_time))

        if i % 100 == 0:
            counter.append(iteration_number)
            loss_history.append(triplet_loss.item())
            plt.plot(counter, loss_history)
            plt.savefig('train_loss.png')
            print("Saved train_loss.png")

        if i % 10 == 0:
            if Config.checkpoint_save_path:
                checkpoint_name = "checkpoint.pt"
                checkpoint_save_path = os.path.join(Config.checkpoint_save_path, checkpoint_name)
                torch.save({
                    'iter': i,
                    'epoch': epoch,
                    'ended_iter': iteration_number
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': triplet_loss.item(),
                }, checkpoint_save_path)
                print("Checkpoint iter: {}, epoch: {}".format(i, epoch))
                print("Checkpoint {} saved in {} successfully".format(checkpoint_name, checkpoint_save_path))

    start_iter = 0

    if not os.path.exists(Config.model_save_path):
        os.mkdir(Config.model_save_path)
    torch.save(net, os.path.joint(Config.model_save_path, "model{}.pt".format(epoch)))

    epoch_elapsed_time = datetime.now() - epoch_start_time
    print("Training epoch time: {}".format(epoch_elapsed_time))

training_elapsed_time = datetime.now() - training_start_time
print("Training done in: {}".format(training_elapsed_time))
