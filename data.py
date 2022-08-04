import random
import os
import os.path as path
from PIL import Image
import numpy as np
import h5py
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

# Ok now this can theoretically load any dataset, we just need the root directory of the batch feed in as the root dir
# The "root" parameter, contrary to the source code, is a list of datasets we want to train on.
# Each dataset is a folder with 3 things: a folder called "dmaps" containing the density maps, a folder called "images" containing the dataset (each data is a folder containing N images), and a text file "meta.txt" containing the metadata of the datasets
# Look at "preprocess.py" for the formatting of the metafile
class listDataset(Dataset):
    def __init__(self, root, root_dir, shuffle = True, shape=None, transform=None, train=False, seen=0, direct=False, batch_size=1, num_workers=4):
        # Inherited variables
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.direct=direct

        # Self defined variables
        self.root_dir = root_dir
        self.X, self.y = [], []
        for dataset in root:
            self.preload_data(dataset)

        # print(len(self.X))
        # print(len(self.y))
        # print(self.X[:2])
        # print(self.y[:2])

        # Shuffle the two arrays simulaneously
        if shuffle:
            # Raise an assertion error if the lists are not the same length, otherwise the shuffling will most probably go wrong
            assert len(self.X) == len(self.y)
            c = list(zip(self.X, self.y))
            random.shuffle(c)
            self.X, self.y = zip(*c)

    # This overloads the len method
    def __len__(self):
        return len(self.X)
    
    # This overloads the indexing method, which is used by pytorch
    def __getitem__(self, index):
        # Make sure the index isnt too big
        assert index <= len(self), 'index range error'
        
        # Load the data
        img, target = load(self.X[index], self.y[index], self.direct)
        return img, target

    # This helps us preload the paths of all the data so we don't need to do this like 500 times during training
    def preload_data(self, data_name):
        img_path = self.root_dir + "datas" + "/" + data_name + "/images"
        dmap_path = self.root_dir + "datas" + "/" + data_name + "/dmaps"

        # with open(self.root_dir + data_name + "/meta.txt") as f:
        #     meta = f.readlines()

        # Now get the images
        for dir in os.listdir(img_path):
            # the dir is really the subfolders/files inside the folder img_path
            # Append the folder which contains exactly one set of data to the self.X list
            path_to_data = img_path + "/" + dir
            self.X.append(path_to_data)
            
            # Now look for the corresponding density map. Also add in the suffix here
            path_to_corr_dmap = dmap_path + "/" + dir + ".npy"

            # Make sure this file exists, otherwise raise an error
            assert path.isfile(path_to_corr_dmap), f"Cannot find corresponding density map ! File missing: {path_to_corr_dmap}"
            self.y.append(path_to_corr_dmap)


# Direct probably means to leave the data alone, don't touch it
# So we won't touch it either since I have no idea what it does
def load(X_path, y_path, direct : bool = False):
    transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    transform2 = transforms.ToTensor()

    # Do something different for the first data, that we initialize the image variable
    images = os.listdir(X_path)
    for i in range(len(images)):
        image_name = images[i]
        if (i == 0):
            img =  np.array(Image.open(X_path + "/" + image_name).convert('RGB'))
            img = transform(img)
            image = img.unsqueeze(dim = 1)
        else:
            new_img =  np.array(Image.open(X_path + "/" + image_name).convert('RGB'))
            new_img = transform(new_img)
            new_image = new_img.unsqueeze(dim = 1)
            image = torch.cat([image, new_image], axis = 1)

    # Now load the density map
    dmap = np.load(y_path)

    # Note: Using PIL instead of OpenCV as in the original code produces an average pixel difference of about 0.002. Insignificant sure but also something noteworthy
    t = Image.fromarray(dmap)
    
    size = (int(np.floor(image.shape[3]/8)), int(np.floor(image.shape[2]/8)))
    t = np.array(t.resize(size, Image.BICUBIC), dtype = dmap.dtype) * 64

    return image, t