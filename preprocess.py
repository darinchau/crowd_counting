from scipy import ndimage
from utility import Printer, searchFile
import cv2
import h5py
import json
import mat4py
import numpy as np
import os
import re
import scipy
import scipy.spatial as ss
import shutil

# number of images per set of data. must be odd number (actually even number should be ok too)
IMAGE_NUM = 3

# This originally says it performs gaussian filter but actually does two things
# coords = data['annotations'], which is a list of list with 2 elements; they are the image coordinates of the position of the humans in the crowd
# First the function creates an image and makes a "one-hot" encoding of the person in the pt2d image
# sigma = 5: Then the function performs Gaussian filtering for each one-hot encoded person
# Gaussian filter is a type of blurrnig with the Gaussian Normal distribution function, where we can get predictable kernels
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

# Little wrapper for mkdir
def mkdir(*paths):
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)

#########################################################
########### This processes the venice dataset ###########
#########################################################
def venice(SOURCE_PATH = './venice', EXPORT_PATH = "./datas/venice"):
    # This following code makes the density maps
    # First initialize the roi and density map files
    # Later in the dictionary we don't actually put the data itself, but we put a tuple (i, data), where i is the index of the file
    # For example, Frame 1 should have index 0, and Frame 42069 should have i = 42068
    # This helps us retrieve the data a bit more easily later

    # Oh btw, roi stands for region of interest
    images, dmaps, roi = {}, {}, {}

    # Get all the density maps which are mat files (for train and test data)
    # List 1 contains tuples of (path to files, file name)
    list1 = searchFile(SOURCE_PATH,'(.*).mat')
    list1.sort()

    list2 = searchFile(SOURCE_PATH,'(.*).jpg')
    list2.sort()

    # Print the progress bar just to make everything fancy
    # Actually this is horrible naming, but in my defence I originally intend it to be called PRINT Events Remaining
    # It just developed from a simple counter with a class wrapper into a progress bar
    printer = Printer(len(list1), "Processing")

    # For each entry in list1:
    #   Load the data
    #   Perform Gaussian filtering on image using the magic function
    #   save the images, roi, and density map in their corresponding dictionaries
    for i in range(len(list1)):   
        # Increment progress bar
        printer.print(i)

        # Load the data (in two dfferent ways)
        try:
            data = mat4py.loadmat(os.path.join(list1[i][0], list1[i][1]))
        except:
            data = h5py.File(os.path.join(list1[i][0],list1[i][1]), 'r')
        
        map = coords_to_density_map(data['annotation'], sigma = 5)

        # The [:-4] indexing chops away the suffix. I know this counts s hardcoding and generally not a good way to do stuff but oh well
        dmaps[list1[i][1][:-4]] = (i, map)
        roi[list1[i][1][:-4]] = (i, data['roi'])

    printer.finish()

    # print("Loading images")

    # Also load the image while we are at it
    for i in range(len(list2)): 
        path_to_img = os.path.join(list2[i][0], list2[i][1])
        image = cv2.imread(path_to_img)
        images[list2[i][1][:-4]] = (i, image)
    
    # We have decided to factor out the second step
    make_data(images, roi, dmaps, EXPORT_PATH)

# We have decided to factor out the second step since we determined that other datasets might also benefit from these methods
# The processed images dictionary has values that looks like (image_name, processed image, density map) and keys are integers
def make_data(images: dict, roi: dict, dmaps: dict, EXPORT_PATH: str):
    # Before we begin, we build the folder structure
    # The folder structure looks like

    # root directory
    #   |
    #   L images
    #   L dmaps
    #   L meta.txt
    #   L anything else

    # The meta txt is not useful right now so it is not generated for now, but we might generate it in the future

    mkdir(EXPORT_PATH, EXPORT_PATH + "/" + "images", EXPORT_PATH + "/" + "dmaps")

    # Initiate the processed_imgs dictionary, and put all the image names into lis
    processed_imgs = {}
    
    lis = list(images.keys())
    lis.sort()

    # The second step is to apply the roi mask. Here we also implicitly perform one extra step, 
    # which is to make the folder structure for the data and check whether each data has a corresponding density map and vice versa
    for i in range(len(lis)):
        # Check if the corresponding density map in list1 exists
        try:
            img_roi = roi[lis[i]][1]
            img_dmap = dmaps[lis[i]][1]
        except:
            # Whoops it probably does not exist. Oh well...
            print(f"Cannot find corresponding density map or roi of {lis[i]}")
            continue
        
        # print(roi[lis[i]][0], dmaps[lis[i]][0], images[lis[i]][0])
        # Get the image and mask it with 
        img = images[lis[i]][1]

        for j in range(3):
            img[:, :, j] = img[:, :, j] * img_roi

        # Store it in dictionary
        # Keys are integers indicating the frame number
        # values are tuples (image name, image, density map)
        processed_imgs[images[lis[i]][0]] = (lis[i], img, img_dmap)
    
    # Get the highest key number
    max_key = np.array(list(processed_imgs.keys()), dtype = int).max()

    # Loop through dictionary to get processed data
    for i in range(max_key):
        # Can 3d evaluates to true if (say IMAGE_NUM = 3) i-1, i, i+1 are all valid frames
        idxs = list(range(i - IMAGE_NUM//2, i + IMAGE_NUM//2 + 1))
        can_3d = np.all([j in list(processed_imgs.keys()) for j in idxs])
        if not can_3d:
            continue

        path_name = EXPORT_PATH + "/" + "images" + "/" + processed_imgs[i][0]
        mkdir(path_name)

        # Save the density map first since that seems to be a bit easier
        dmap_path = EXPORT_PATH + "/" + "dmaps" + "/" + processed_imgs[i][0] + ".npy"
        np.save(dmap_path, processed_imgs[i][2])

        for j in idxs:
            img_path = path_name + "/" + processed_imgs[j][0] + ".jpg"
            cv2.imwrite(img_path, processed_imgs[i][1])


# We need to achieve 3 things
# Make the densiy maps and put it somewhere in the unity folder
# Mask the images with the roi
# Put the 3d dataset somewhere and export the paths to train.json

def process_string():
    # This processes the txt data file into tuples of (2, Number of people) numpy arrays
    pass

def unity():
    pass


# Entry point
if __name__ == "__main__":
    venice(SOURCE_PATH="./fakevenice", EXPORT_PATH="./datas/fakevenice")
    # venice(SOURCE_PATH="./venice/train_data", EXPORT_PATH="./datas/Venice Batch 1")
    # venice(SOURCE_PATH="./venice/test_data", EXPORT_PATH="./datas/Venice Batch 2")