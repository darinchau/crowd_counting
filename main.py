import os
from phnet import PHNet
import torch
import torch.nn as nn
from torchvision import transforms
import argparse
import json
import data as dataset
import time
from utility import History, save_checkpoint
import random
from PIL import Image
import numpy as np
import h5py
import cv2
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from utility import searchFile

# Args namespace holds all variables about the training


def main(args):
    global device, root

    # Learning rate and initial learning rate. We use learning rate decay to adjust for larger epochs to avoid overfitting
    args.original_lr    = 1e-6
    args.lr             = 1e-6
    args.decay          = 5*1e-4

    # We have start epoch to account for checkpoint system
    args.start_epoch    = 0
    args.epochs         = 400
    args.steps          = [-1,1,100,150]
    args.scales         = [1,1,1,1]

    # Multiprocessing good guy
    args.workers        = 4
    args.logname        = "logs.txt"

    root = args.user_dir

    # Load list of file for list of files to train and test
    with open(root + args.train_json, 'r') as outfile:
        train_list = json.load(outfile)

    with open(root + args.test_json, 'r') as outfile:
        val_list = json.load(outfile)

    # Flush log file
    with open(root + "" + args.logname, "w") as f:
        pass


    # Set up gpu if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Make model
    model = PHNet()
    model.to(device)

    criterion = nn.MSELoss(reduction='sum').to(device)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.decay)

    # Sets up parallel computing
    model = DataParallel_withLoss(model, criterion)

    # Loads training data
    train_dataloader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                       shuffle = True,
                       transform = transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),
                       train = True,
                       gt_code = args.gt_code,
                       batch_size = args.batch_size,
                       num_workers = args.workers),
        batch_size = args.batch_size)

    test_dataloader = torch.utils.data.DataLoader(
        dataset.listDataset(val_list,
                    shuffle=False,
                    gt_code = args.gt_code,
                    transform=transforms.Compose([
                        transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
                    ])
                        ,train=False),
        batch_size = args.batch_size)
    
    # Sets up training history
    history = History(len(train_dataloader.dataset), args.epochs, progress_bar = args.progress_bar)

    # Tries to load checkpoint. Everything is updated by reference
    Load_Checkpoint(args.pre, history, model, optimizer)

    while history.current_epoch <= history.epochs:
        # Adjust learning rate according to decay
        adjust_learning_rate(optimizer, history.current_epoch)

        # Training loop
        train(train_dataloader, model, optimizer, history, args.batch_size)

        # Testing loop
        test(test_dataloader, model, args.batch_size, history)
        
        with open(root + "" + args.logname, "a") as f:
            f.write("epoch " + str(history.current_epoch) + "  mae: " +str(float(history.history["val_acc"][-1])))
            f.write("\n")

        # Saves checkpoint
        save_checkpoint({
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            'history': history,
        }, history.is_Best, args.task, epoch = history.current_epoch, path= root + "ckpt/"+args.task)


# Main training loop
def train(dataloader, model, optimizer, history, batch_size):
    history.new_epoch()    
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Prepares the data: move to cuda if available
        X, y = X.to(device), y.to(device)
        X = X.type(torch.FloatTensor)
        y = y.type(torch.FloatTensor).unsqueeze(1).to(device)

        # Fwprop
        loss, output = model(y, X)

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mae = GAME(output.data, y, 0) / batch_size

        # Update the result
        history.increment(batch * len(X), loss.item(), mae.item())


def test(dataloader, model, batch_size, history):
    model.eval()
    mae = 0
    
    with torch.no_grad():
        for batch, (X, y) in enumerate(dataloader):
            X = X.to(device)
            X = X.type(torch.FloatTensor)
            y = y.type(torch.FloatTensor).unsqueeze(1).to(device)
            loss , output = model(y, X)
            mae += GAME(output.data, y, 0)
    
    # mae /= number of data
    mae = mae/len(dataloader)/batch_size
    history.validate(loss.item(), mae.item())
    return mae


# Something like an evaluation metric?
def GAME(img, target, level = 1):
    batch_size = img.shape[0]
    w, h = img.shape[2], img.shape[3]
    w, h = w//(level + 1), h//(level + 1)
    game = 0
    for batch in range(batch_size):
        game += abs(img[batch,:,:,:].sum() - target[batch,:,:,:].sum())
    return game

# Try to load a pretrained model in "p". Returns the model if something is found, otherwise return none
def Load_Checkpoint(p, history, model, optimizer):
    if p:
        ## TODO remember to change checkpoint structure
        if os.path.isfile(p):
            print("=> loading checkpoint '{}'".format(p))
            checkpoint = torch.load(p)
            history.load(checkpoint['history'])
            model_dict = model.state_dict()
            pretrained_dict = {k : v for k, v in checkpoint['state_dict'].items() if k in model_dict}
            model_dict.update(pretrained_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded checkpoint '{p}' (epoch {history.current_epoch})")
        
        print("=> no checkpoint found at '{}'".format(p))


# Sets up parallel computing
def DataParallel_withLoss(model, loss,**kwargs):
    model=FullModel(model, loss)
    if 'device_ids' in kwargs.keys():
        device_ids=kwargs['device_ids']
    else:
        device_ids=None
    
    if 'output_device' in kwargs.keys():
        output_device=kwargs['output_device']
    else:
        output_device=None
    
    if 'cuda' in kwargs.keys():
        cudaID=kwargs['cuda']
        model=torch.nn.DataParallel(model, device_ids=device_ids, output_device=output_device).cuda(cudaID)
    else:
        model=torch.nn.DataParallel(model, device_ids=device_ids, output_device=output_device).to(device)
    
    return model


class FullModel(nn.Module):
    def __init__(self, model, loss):
        super(FullModel, self).__init__()
        self.model = model
        self.loss = loss

    def forward(self, targets, *inputs):
        outputs = self.model(*inputs)
        #print(outputs.shape)
        loss = self.loss(outputs, targets)
        return torch.unsqueeze(loss, 0), outputs


# Adjust learning rate according to decay
def adjust_learning_rate(optimizer, epoch):
    args.lr = args.original_lr
    for i in range(len(args.steps)):
        scale = args.scales[i] if i < len(args.scales) else 1
        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr


if __name__ == "__main__":
    # python model/train.py /dataset/Venice/train_data.json /dataset/Venice/test_data.json
    # Environment variables
    parser = argparse.ArgumentParser(description='UROP 1100')
    parser.add_argument('--train_json', metavar='TRAIN', help='path to train json', default="jsons/train3.json")
    parser.add_argument('--test_json', metavar='TEST', help='path to test json', default="jsons/test3.json")
    parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None,type=str, help='path to the pretrained model')
    parser.add_argument('--batch_size', '-bs', metavar='BATCHSIZE' ,type=int, help='batch size', default=2)
    parser.add_argument('--gpu',metavar='GPU', type=str, help='GPU id to use.', default="5")
    parser.add_argument('--task',metavar='TASK', type=str, help='task id to use.', default="1")
    parser.add_argument('--gt_code', metavar='GT_NUMBER' ,type=str, help='ground truth dataset number', default='4896')
    parser.add_argument('--progress_bar', metavar='PBAR' ,type=bool, help='Whether to use progress bar or not', default=False)
    parser.add_argument('--user_dir', metavar="USERDIR", type=str, default="./")

    args = parser.parse_args()

    main(args)