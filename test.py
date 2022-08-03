from utility import History
import random
import time
from scipy.io import loadmat
from utility import searchFile
from preprocess import SOURCE_PATH
import mat4py
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import cv2

def test1():
    """Test progress bar"""
    epochs = 3
    size = 100
    history = History(size, epochs, progress_bar = False)
    for i in range(epochs):
        history.new_epoch()
        for j in range(size):
            history.increment(j, random.random(), random.random())
            time.sleep(.000001)
        history.validate(random.random(), random.random())

    # Local machine: python main.py --progress_bar=TRUE
    # TACC: python ${TACC_WORKDIR}/main.py --user_dir=${TACC_USERDIR}/

# Exploring the venice dataset
def test2(data_idx):
    list1 = searchFile(SOURCE_PATH,'(.*).mat')
    list1.sort()
    try:
        data = mat4py.loadmat(os.path.join(list1[data_idx][0], list1[data_idx][1]))
    except:
        data = h5py.File(os.path.join(list1[data_idx][0],list1[data_idx][1]), 'r')
    # print(type(data))         # Dictionary
    # print(*data.keys())       # roi homograph annotation, each value is  a list

    # for k in data.keys():
    #     print(k)
    #     print(len(data[k]), data[k][0])

    # ROI seems to be the region of interest
    # A list of list of 1s and 0s, like a mask

    # roi = np.array(data['roi'])
    # plt.figure()
    # plt.imshow(roi, cmap = 'binary')
    # plt.show()

    # Yup

    # homograph seems like they are never used

    # annotation are a list of coordinates about the position of the humans
    pass

def plot(img, cmap = "viridis"):
    plt.figure()
    plt.imshow(img, cmap)
    plt.show()


# Exploring data['annotations']
def test3():
    list1 = searchFile(SOURCE_PATH,'(.*).mat')
    list1.sort()
    l = max(len(list1), 1)

    for i in range(l):   
        try:
            data = mat4py.loadmat(os.path.join(list1[i][0], list1[i][1]))
        except:
            data = h5py.File(os.path.join(list1[i][0],list1[i][1]), 'r')
        map = np.array(data['annotation'])
        plot(map)
        print(map.max(), map.min(), map.dtype, map.shape)
        map = gaussian_filter_density_new(map, 5)
        plot(map)
        print(map.max(), map.min(), map.dtype, map.shape)


# Performs Gaussian filter on an image. Gaussian filter is a type of blurrnig with the Gaussian distribution function, where we can get predefined kernels
def gaussian_filter_density_new(gt, sigma=5):
    sha = (720, 1280)
    # Initialize the density map
    density = np.zeros(sha, dtype=np.float32)

    # For each coordinate point
    for i in range(len(gt)):
        pt2d = np.zeros(sha, dtype=np.float32)
        try:
            pt2d[gt[i][1], gt[i][0]] = 1.
        except:
            pt2d[gt[i][1]-1, gt[i][0]-1] = 1.
        density += ndimage.gaussian_filter(pt2d, sigma, mode='constant')
    return density

# Exploring the annotations
def test4():
    list1 = searchFile(SOURCE_PATH,'(.*).mat')
    list1.sort()
    for data_idx in range(100):
        try:
            data = mat4py.loadmat(os.path.join(list1[data_idx][0], list1[data_idx][1]))
        except:
            data = h5py.File(os.path.join(list1[data_idx][0],list1[data_idx][1]), 'r')
        ann = data['annotation']
        print(len(ann))

if __name__ == "__main__":
    test4()