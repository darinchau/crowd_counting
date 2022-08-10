import os
import time
import torch
import numpy as np
import h5py
import shutil
import re

hehehaha = "hehehaha"

# Alternative, simpler implementation
# Simple progress bar
class ProgressBar:
    def __init__(self, total_fr, task_name, len_bar = 40, pad = 10):
        print(task_name)
        self.fr = 0
        self.total_fr = total_fr
        self.len_bar = len_bar
        self.last = ""
        self.__enabled = True
        self.pad = pad
        self.increment(0)
        
    def finish(self):
        self.disable()
        print()

    def increment(self, n = 1):
        if not self.__enabled:
            return

        self.fr += n

        ratio = round(self.fr/self.total_fr * self.len_bar)
        st = "[" + ratio * "=" + (self.len_bar - ratio) * " " + "]  " + str(self.fr) + "/" + str(self.total_fr) + " " * self.pad
        print("\b" * len(self.last) + st, end = "", flush = True)
        self.t = 0
        self.last = st

    def disable(self):
        self.__enabled = False


class Timer:
    @staticmethod
    def __init__(f, *args, **kwargs):
        t1 = time.time()
        out = f(*args, **kwargs)
        t2 = time.time()
        return round(t2 - t1, 5), out


_callbacks = {}

class Event():
    @staticmethod
    def on(event_name, f):
        if event_name in _callbacks and f in _callbacks[event_name]:
            return

        _callbacks[event_name] = _callbacks.get(event_name, []) + [f]

    # Emit is not really an event but just calling the functions in callback one by one
    @staticmethod
    def emit(event_name, *data):
        global stop_threads
        for f in _callbacks.get(event_name, []):
            f(*data)
            if stop_threads:
                break

    @staticmethod
    def off(event_name, f):
        try:
            _callbacks.get(event_name, []).remove(f)
        except ValueError:
            pass

class History:
    def __init__(self, total_data, epochs, seed = 42069):
        # (Generate and) save the seed
        if seed is None:
            seed = np.random.randint(2147483647)
        torch.manual_seed(seed)
        self.initial_seed = seed

        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

        self.total_data = total_data

        self.best_val_loss = 1e99
        self.best_val_acc = 0
        self.best_epoch = -1
        self.current_epoch = 0
        self.epochs = epochs
        
        # Is_best is a temporary flag that gets set to true if the best epoch is updated
        self.is_Best = False

    def new_epoch(self):        
        self.is_Best = False
        self.current_epoch += 1
        self.correct = 0
    
    def increment(self, train_loss, train_acc):
        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
    
    def validate(self, val_loss, val_acc):
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            self.best_epoch = self.current_epoch
            self.is_Best = True
        
        print(f"Validation: loss = {val_loss}, MAE = {val_acc}")


    def load(self, src):
        self.initial_seed = src.initial_seed

        self.history = src.history

        self.best_val_loss = src.best_val_loss
        self.best_val_acc = src.best_val_acc
        self.best_epoch = src.best_epoch
        self.current_epoch = src.current_epoch


def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():        
            param = torch.from_numpy(np.asarray(h5f[k]))         
            v.copy_(param)


def save_checkpoint(state, is_best,task_id, epoch, filename='checkpoint.pth', path = "."):
    if not os.path.exists(path):
            os.makedirs(path)
    if is_best:
        torch.save(state, path + "/" + task_id + "_" + str(epoch)+ filename)
        shutil.copyfile(path + "/" + task_id + "_" + str(epoch)+ filename, path + "/" + task_id+'model_best.pth')


# Search for all files within a folder and all its subfolders
def searchFile(pathname,filename):
    matchedFile = []
    for root, dirs, files in os.walk(pathname):
        for file in files:
            if re.match(filename,file):
                matchedFile.append((root,file))
    return matchedFile