import os
import mat4py
import h5py
import numpy as np
import scipy
import scipy.spatial as ss
from scipy import ndimage
import re
import shutil
import cv2
import json
from utility import Printer, searchFile


SOURCE_PATH = './venice'
IMG_ROI_PATH = './venice/img_roi'
ROI_PATH = './venice/density_map_init'

# number of images per set of data (image/frame etc)
IMAGE_NUM = 3

EXPORT_PATH = './venice/ablation' + str(IMAGE_NUM)

# This originally says it performs gaussian filter but actually does two things
# coords = data['annotations'], which is a list of list with 2 elements; they are the image coordinates of the position of the humans in the crowd
# First the function creates an image and makes a "one-hot" encoding of the crowd in the pt2d image
# sigma = 5: Then the function performs Gaussian filtering for each one-hot encoded person
# Gaussian filter is a type of blurrnig with the Gaussian distribution function, where we can get predefined kernels
def coords_to_density_map(coords, sigma=5):
    sha = (720, 1280)
    density = np.zeros(sha, dtype=np.float32)
    for i in range(len(coords)):
        pt2d = np.zeros(sha, dtype=np.float32)
        try:
            pt2d[coords[i][1], coords[i][0]] = 1.
        except:
            pt2d[coords[i][1]-1, coords[i][0]-1] = 1.
        density += ndimage.gaussian_filter(pt2d, sigma, mode='constant')
    return density

def make_density_map():
    # Get all the density maps which are mat files (for train and test data)
    # List 1 contains tuples of (path to files, file name)
    list1 = searchFile(SOURCE_PATH,'(.*).mat')
    list1.sort()

    # Print the progress bar just to make everything fancy
    printer = Printer(len(list1), "Processing")

    # For each entry in list1:
    #   Load the data
    #   Perform Gaussian filtering on image using the magic function
    for i in range(len(list1)):   
        # Increment progress bar
        printer.print(i)

        try:
            data = mat4py.loadmat(os.path.join(list1[i][0], list1[i][1]))
        except:
            data = h5py.File(os.path.join(list1[i][0],list1[i][1]), 'r')
        map = coords_to_density_map(data['annotation'], 5)
        path = os.path.join(ROI_PATH, list1[i][1]).replace('.mat','.h5')
        if not os.path.exists(ROI_PATH):
                os.makedirs(ROI_PATH)
        with h5py.File(path, 'w') as hf:
                hf['density'] = map
                hf['roi'] = data['roi']
    
    printer.finish()

def mycopyfile(srcfile,dstfile, verbose = False):
    if not os.path.isfile(srcfile):
        print("not exist!"%(srcfile))
    else:
        fpath, fname=os.path.split(dstfile)   
        if not os.path.exists(fpath):
            os.makedirs(fpath)               
        shutil.copyfile(srcfile,dstfile)     
        if verbose: print("copy %s -> %s"%( srcfile,dstfile))

# ROI = Region of interest
def process_roi():
    # Get all jpg files. This is frames of train/test datas
    list1 = searchFile(SOURCE_PATH, '(.*).jpg')

    # Get all the h5 files with the same name
    # put on the roi mask
    # Save a new image
    for i in range(len(list1)):
        roi_path = os.path.join(ROI_PATH, list1[i][1].replace('.jpg', '.h5'))
        data = h5py.File(roi_path, 'r')
        source = os.path.join(list1[i][0], list1[i][1])
        target = os.path.join(IMG_ROI_PATH, list1[i][1])
        mycopyfile(source, target)
        img = cv2.imread(target)
        for i in range(3):
            img[:,:,i] = img[:,:,i] * data['roi']
        cv2.imwrite(target, img)

# Returns true if the splitting of the frames are normal  
def normal(ind1, ind2, list1, verbose = False):
    if ind1 < 0:
        return False
    if list1[ind1][1].split('_')[0] != list1[ind2][1].split('_')[0]:
        return False
    i1 = int(list1[ind1][1].split('.')[0].split('_')[1])
    i2 = int(list1[ind2][1].split('.')[0].split('_')[1])
    if (i2-i1) > 60 * (ind2-ind1+1):
        if verbose: print('{} lost more than one frame to {}'.format(list1[ind1][1], list1[ind2][1]))
        return False
    return True


# Make copies of data???? Not sure what this is doing.
def make_3d_dataset():
    train_json = []
    test_json = []

    list1 = searchFile(IMG_ROI_PATH, '(.*).jpg')
    list1.sort()

    # For each image do a for loop to copy image files
    for index in range(len(list1)):
        for place in range(IMAGE_NUM - 1, -1, -1):
            # print(index, place, normal(index - place, index, list1))
            if normal(index - place, index, list1):
                source = os.path.join(IMG_ROI_PATH, list1[index - place][1])
                target = os.path.join(EXPORT_PATH, list1[index][1], list1[index - place][1])
                mycopyfile(source, target)
            else:
                source = os.path.join(IMG_ROI_PATH, list1[index][1])
                target = os.path.join(EXPORT_PATH, list1[index][1], list1[index][1].split('.')[0] + str(place)+'.jpg')
                mycopyfile(source, target)
        
        # Export it to either test json or train json
        if list1[index][1].split('_')[0] == '4896':
            train_json.append(os.path.join(EXPORT_PATH, list1[index][1]))
        else:
            test_json.append(os.path.join(EXPORT_PATH, list1[index][1]))

    # Create the path if it does not already exist - this prevents error in the next step
    for path in ['./jsons/train', './jsons/test']:
        if not os.path.exists(path):
            os.makedirs(path)

    # Make the json file
    with open('./jsons/train' + str(IMAGE_NUM) + '.json', 'a', encoding='utf-8') as json_file:
        json.dump(train_json, json_file, indent=1)
        print(len(train_json))

    with open('./jsons/test' + str(IMAGE_NUM) + '.json', 'a', encoding='utf-8') as json_file:
        json.dump(test_json, json_file, indent=1)
        print(len(test_json))

def venice():
    make_density_map()
    process_roi()
    make_3d_dataset()
    
if __name__ == "__main__":
    venice()