import os
import time
import torch
import numpy as np
import h5py
import shutil
import re

hehehaha = "hehehaha"

# Progress bar
# Initialize printer before use
# Then update the printer using the print function in every iteration
# Finally just let it kill itself or call printer.finish() to release the print flush
class Printer:
    def __init__(self, total_fr, description = " ", print_every = 0.2, len_bar = 40, pad = 10, enabled = True):
        self.total_fr = total_fr
        self.print_every = print_every
        self.len_bar = len_bar
        self.last = ""
        self.__enabled = enabled
        self.description = description
        if description[-1] != " ":
            self.description += " "
        self.t = 0
        self.last_t = time.time()
        self.last_desc = ""
        
        # Do padding to avoid trailing zeros
        self.pad = pad

        self.print(0, "", True)
    
    def finish(self):
        if self.__enabled:
            self.print(self.total_fr, self.last_desc)
            self.__enabled = False
            print()
    
    def print_in(self, *args, **kwargs):
        print("\b" * len(self.last), end = "")
        print(*args, **kwargs)
        if "end" in kwargs.keys() and kwargs["end"] != "\n":
            print()
        print(self.last, end = "", flush = True)

    
    def print(self, fr, description = "data", force = False):
        if not self.__enabled:
            return
        
        # Increment time
        # If time > print_every_time_interval or last one or force out:
        #   do printing na
        
        t = time.time()
        self.t += t - self.last_t
        self.last_t = t
        self.last_desc = description
        
        if self.t >= self.print_every or fr >= self.total_fr - 1 or force:
            ratio = round((fr + 1)/self.total_fr * self.len_bar)
            st = self.description + description + ": [" + ratio * "=" + (self.len_bar - ratio) * " " + "]  " + str(fr) + "/" + str(self.total_fr) + " " * self.pad
            print("\b" * len(self.last) + st, end = "", flush = True)
            self.t = 0
            self.last = st
        
        self.last_desc = description

    def __del__(self):
        self.finish()

    def disable(self):
        self.__enabled = False

class Timer:
    @staticmethod
    def __init__(self, f, *args, **kwargs):
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
    def __init__(self, total_data, epochs, seed = 42069, progress_bar = True):
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
        
        # Makes a kill switch for the printer so its easier to have some fun
        self.verbose = progress_bar

    def new_epoch(self):        
        self.is_Best = False
        self.current_epoch += 1
        description = f"Current epoch: {self.current_epoch}, "
        self.printer = Printer(self.total_data, description = description, enabled = self.verbose)
        
        self.correct = 0
    
    def increment(self, current_progress, train_loss, train_acc):
        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
        
        # len(X) is really batch size. Makes the fancy progress bar thing
        description = f"loss: {round(train_loss, 5)}, acc: {round(train_acc, 5)} "
        self.printer.print(current_progress, description)
    
    def validate(self, val_loss, val_acc):
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)

        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            self.best_epoch = self.current_epoch
            self.is_Best = True
        
        self.printer.finish()
        print(f"Validation: loss = {val_loss}, MAE = {val_acc}")


    def load(self, src):
        self.initial_seed = src.initial_seed

        self.history = src.history

        self.best_val_loss = src.best_val_loss
        self.best_val_acc = src.best_val_acc
        self.best_epoch = src.best_epoch
        self.current_epoch = src.current_epoch

    def print(self, *args, **kwargs):
        self.printer.print_in(*args, **kwargs)


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