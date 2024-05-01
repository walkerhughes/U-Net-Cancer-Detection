import time
import argparse 
from tqdm import tqdm 
import numpy as np 

import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader

from model.dataset import CancerDataset
from model.model import UNetCancerDetection

from utils import plot_loss_and_accuracy, plot_predictions


if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description="Train UNet Model for Cancer Detection")
    parser.add_argument('-epochs', type = int, default = 3, help='Number of epochs to train')
    parser.add_argument('-validation_image', type = int, default = 172, help='Validation image to plot predicted segmentation')
    args = parser.parse_args()

    print(f"\nModel will train for {args.epochs} epochs and show predicted image segmentation for image {args.validation_image}.\n\n")

    # init datasets, dataloaders and model 
    train_data = CancerDataset("tmp/cancer", train = True)
    validation_data = CancerDataset("tmp/cancer", train = False)

    # read in data and set up data loaders 
    train_loader = DataLoader(train_data, batch_size = 4, pin_memory = True, shuffle = True)
    valid_loader = DataLoader(validation_data, batch_size = 4, pin_memory = True)

    # init model and call cuda 
    model = UNetCancerDetection(None) # UNetCancerDetection(train_data) 
    model = model.cuda()

    # define objective and optimizer 
    objective = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-4) 

    # Your plotting code here
    train_losses = []
    val_losses = []
    train_accuracy = []
    val_accuracy = []

    # Run training / validation loops
    start = time.time()

    for epoch in range(args.epochs): 

        loop = tqdm(total = len(train_loader), position = 0, leave = False) 

        for batch, (x, y_truth) in enumerate(train_loader): 

            # x, y_truth = x.cuda(async = True), y_truth.cuda(async = True)
            x, y_truth = x.cuda(), y_truth.cuda()

            optimizer.zero_grad()
            y_hat = model(x)  
            loss = objective(y_hat, y_truth.long())
            loss.backward()

            current_time = time.time() - start 
            train_losses.append((current_time, loss.item()))

            # compute accuracy and append to list above 
            accuracy = ((torch.softmax(y_hat, 1).argmax(1) == y_truth).float()).mean()
            train_accuracy.append((current_time, accuracy)) 
            loop.set_description("epoch: {} loss: {:.4f} accuracy: {:.4}".format(epoch, loss.item(), accuracy))
            loop.update()
            optimizer.step()

            # if we are to check validation set 
            if batch % 100 == 0: 

                # compute loss, accuracy 
                val = np.mean([objective(model(x.cuda()), y.long().cuda()).item() for x, y in valid_loader])
                val_losses.append((current_time, val))

                # take accuracy, comparing to true classes and taking the mean of vector of results 
                v_accuracy = torch.mean((torch.stack([((model(x.cuda()).argmax(1) == y.cuda()).float()).mean() for x, y in valid_loader])))
                val_accuracy.append((current_time, v_accuracy))
                
        loop.close()

    plot_loss_and_accuracy(train_losses, train_accuracy, val_losses, val_accuracy)
    plot_predictions(model, validation_data, args.validation_image)