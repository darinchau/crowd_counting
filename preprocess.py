from checkframe import check_bounding_box, check_frame
from scipy import ndimage
from multiprocessing import Process, Queue
from unitysrc import process_unity_data
from utility import searchFile, ProgressBar
import cv2
import h5py
import json
import mat4py
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import scipy
import scipy.io
import scipy.spatial as ss
import shutil
import sys

# number of images per set of data. must be odd number (actually even number should be ok too)
IMAGE_NUM = 3

# This originally says it performs gaussian filter but actually does two things
# coords = data['annotations'], which is a list of list with 2 elements; they are the image coordinates of the position of the humans in the crowd
# First the function creates an image and makes a "one-hot" encoding of the person in the pt2d image
# sigma = 5: Then the function performs Gaussian filtering for each one-hot encoded person
# Gaussian filter is a type of blurrnig with the Gaussian Normal distribution function, where we can get predictable kernels
# sha is the image dimensions
def coords_to_density_map(coords, img_size=(720, 1280), sigma=5):
    # sha is target output size
    sha = (720, 1280)
    # Initialize density map
    density = np.zeros(sha, dtype=np.float32)

    # Loop through every coordinates
    for i in range(len(coords)):
        # Tranlate coordinates from (img size) into (actual size)
        x = coords[i][1] * sha[1] // img_size[1]
        y = coords[i][0] * sha[0] // img_size[0]
        pt2d = np.zeros(sha, dtype=np.float32)
        try:
            pt2d[x, y] = 1.
        except:
            pt2d[x-1, y-1] = 1.
        density += ndimage.gaussian_filter(pt2d, sigma, mode='constant')
    return density

# Little wrapper for mkdir
def mkdir(*paths):
    for path in paths:
        if not os.path.isdir(path):
            os.makedirs(path)

# I think the most fascinating thing about the cv2 resize is it can handle 2 dimensional arrays
# size is (cols, rows)
def try_resize(img, size = (1280, 720)):
    img = cv2.resize(img, size, interpolation = cv2.INTER_CUBIC)
    return img

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
    
    # Loop through dictionary to get processed data
    for i in processed_imgs.keys():
        # Can 3d evaluates to true if (say IMAGE_NUM = 3) i-2, i-1, i are all valid frames
        idxs = list(range(i - IMAGE_NUM + 1, i + 1))
        can_3d = np.all([j in list(processed_imgs.keys()) for j in idxs])
        if not can_3d:
            continue

        # Make directories for image base path
        path_name = EXPORT_PATH + "/" + "images" + "/" + processed_imgs[i][0]
        mkdir(path_name)

        # Save the density map first since that seems to be a bit easier
        dmap_path = EXPORT_PATH + "/" + "dmaps" + "/" + processed_imgs[i][0] + ".npy"
        np.save(dmap_path, processed_imgs[i][2])

        # Also save a image of the dmap for easy visualization
        dmap_img_path = dmap_path[:-4] + ".jpg"
        mpimg.imsave(dmap_img_path, processed_imgs[i][2])

        # Save he images
        for j in idxs:
            img_path = path_name + "/" + processed_imgs[j][0] + ".jpg"
            cv2.imwrite(img_path, processed_imgs[j][1])

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
    pbar = ProgressBar(len(list1), "Processing data for " + SOURCE_PATH[2:])

    # For each entry in list1:
    #   Load the data
    #   Perform Gaussian filtering on image using the magic function
    #   save the images, roi, and density map in their corresponding dictionaries
    for i in range(len(list1)):   
        # Increment progress bar
        pbar.increment()

        # Load the data (in two dfferent ways)
        try:
            data = mat4py.loadmat(os.path.join(list1[i][0], list1[i][1]))
        except:
            data = h5py.File(os.path.join(list1[i][0],list1[i][1]), 'r')
        
        map = coords_to_density_map(data['annotation'], sigma = 5)

        frame_idx = int(list1[i][1][7:-4])

        # The [:-4] indexing chops away the suffix. I know this counts s hardcoding and generally not a good way to do stuff but oh well
        dmaps[list1[i][1][:-4]] = (frame_idx, map)
        roi[list1[i][1][:-4]] = (frame_idx, data['roi'])

    pbar.finish()

    # print("Loading images")

    # Also load the image while we are at it
    for i in range(len(list2)): 
        path_to_img = os.path.join(list2[i][0], list2[i][1])
        image = cv2.imread(path_to_img)
        frame_idx = int(list2[i][1][7:-4])
        images[list2[i][1][:-4]] = (frame_idx, image)
    
    # We have decided to factor out the second step
    make_data(images, roi, dmaps, EXPORT_PATH)

# We wrote another function since my computer is not sufficient to handle the big stuff
# So there would be fancy schmancy stuff like multiprocessing etc
########################################################
########### This processes the unity dataset ###########
########################################################
def unity(SOURCE_PATH, EXPORT_PATH, _check_frame = False):
    # First read the data file
    data = ""
    with open(SOURCE_PATH + "/" + "data.txt") as f:
        data = f.read()

    data_dict = process_unity_data(data)

    # We read all the paths of images
    dirs = os.listdir(SOURCE_PATH)
    dirs.sort()
    
    # dmaps will be done via multiprocessing
    images, roi, dmaps = {}, {}, {}

    # Preload the roi since it is the same for every image
    roi_path = SOURCE_PATH + "/" + "roi.png"
    if os.path.isfile(roi_path):
        bnw_roi = cv2.imread(roi_path)[:,:,0]
        bnw_roi[bnw_roi > 0] = 1
        roi_img = try_resize(np.array(bnw_roi, dtype = np.uint8))
    else:
        roi_img = np.zeros((720, 1280), dtype = float) + 1

    # Progrss bar
    pbar = ProgressBar(len(dirs), "Processing data " + SOURCE_PATH[8:])
    for i in range(len(dirs)):
        pbar.increment()
        dir = dirs[i]

        # Skip the data.txt and roi.png
        if dir == "data.txt" or dir[:3] == "roi":
            continue
        
        # Make sure the pictures have .png suffix
        assert dir[-4:] == ".png"
        
        # Since we are exporting in png format we need to change it to jpg format
        # We sort of achieve this by chopping the alpha channel, multiply everythign by 256, and floor it, and finally convert it to unsigned 8 bit integer format
        # This array will not be something that you would load from a jpg but it resembles the format of it
        # So for the purposes of our training it will be good enough
        # CV2 loads stuff in bgr so we have to do something about it

        # No need to change to rgb since we are saving the iamges with bgr
        image = cv2.imread(SOURCE_PATH + "/" + dir)

        if image.max() < 2:
            image = np.round(image * 255)
        if image.shape[-1] == 4:
            image = image[:, :, :-1]
        
        img = np.array(image, dtype = np.uint8)
        
        # This will serve as the key of our dictionaries
        img_name = dir[:-4]

        # Get the frame number as appeared on the image name
        frame_idx = int(img_name.split("Frame ")[1])

        # Get the ith frame data, -1 means the last index which returns the frame data (refer to above) of that frame
        frame_i_data = data_dict[frame_idx][-1]

        # Extact the coordinate info
        c = [(frame_i_data[k][0], frame_i_data[k][1]) for k in range(len(frame_i_data))]
        
        
        # Turn this into a numpy array to make the density map
        c = np.array(c, dtype = int)
        
        
        # If the check frame flag is set to true
        if _check_frame:
            # Turn down the brightness and make a white dot on all the specified coordinates
            check_frame(c, img)
            # Also check the bounding boxes while we are at it
            bbs = [frame_i_data[k][3:] for k in range(len(frame_i_data))]
            bbs = np.array(bbs, dtype = int)
            check_bounding_box(img, bbs)
            

        dmap = coords_to_density_map(c, img_size = img.shape[:-1], sigma = 5)

        # print(dmap.shape, c.shape, np.sum(dmap))
        # plt.figure()
        # plt.imshow(dmap)
        # plt.show()

        # Resize image
        img = try_resize(img)

        # Same format as above: keys = image name, values = tuple(frame index, image/roi/dmap)
        images[img_name] = (frame_idx, img)
        roi[img_name] = (frame_idx, roi_img)
        dmaps[img_name] = (frame_idx, dmap)

    pbar.finish()

    # Offload our work to other functions like a true lazy computer programmer 
    make_data(images, roi, dmaps, EXPORT_PATH)



############################################################
########### This processes the CrowdFlow dataset ###########
############################################################
# src is source path, export is export path
# crowdflow is a little bit different with its folder structure so src should point to the folder with the README.txt
# Then ds_name (dataset name) is something like "IM01", "IM02" etc
# In the google drive we provided rois for the static frames. Feel free to download those and put te 5 roi masks in an roi folder
def crowdflow(src, export, ds_name, _check_frame = False):
    # Set the base path of the images
    base_path = src + "/" + "TUBCrowdFlow"
    # Set the image path
    img_path = base_path + "/" + "images" + "/" + ds_name
    # Set the roi path. If there is no roi folder we will default to no roi
    roi_path = base_path + "/" + "roi" + "/" + ds_name + ".png"
    # Set the ground truth path
    gt_path = base_path + "/" + "gt_trajectories" + "/" + ds_name + "/" + "personTrajectories.mat"

    # Load the ground truths first. They are dictionaries with two different naming conventions. So we try to load both
    gt = scipy.io.loadmat(gt_path)
    try:
        coords = gt["matTrajects"]
    except:
        coords = gt["matTrajectsCopy"]
    
    # The coords is (un)conveniently provided in a numpy array of shape (number of humans???, frame count x 2)
    # And it turns out the frame count x 2 is actually the x and y coordinates concatenated in the same dimension
    # We fix this by reshape
    coords = coords.reshape(-1, coords.shape[1]//2, 2)

    # Now load the roi. If the roi path is really a file, then load the roi, otherwise make a white image
    if os.path.isfile(roi_path):
        bnw_roi = cv2.imread(roi_path)[:,:,0]
        bnw_roi[bnw_roi > 0] = 1
        roi = try_resize(np.array(bnw_roi, dtype = np.uint8))
    else:
        roi = np.zeros((720, 1280), dtype = float) + 1

    # Now we get all the images. Paths first
    imgs_path = os.listdir(img_path)
    imgs_path.sort()

    # Initiate the dictionaries
    imgs, rois, dmaps = {}, {}, {}

    # Progress bar
    p = ProgressBar(len(imgs_path), "Processing data for " + ds_name)

    # Loop through every image path
    for i in range(len(imgs_path)):
        dir = imgs_path[i]

        # Progress bar
        p.increment()

        # Make sure the pictures have .png suffix
        assert dir[-4:] == ".png"

        # No need to change to rgb since we are saving the iamges with bgr
        image = cv2.imread(img_path + "/" + dir)

        if image.max() < 2:
            image = np.round(image * 255)
        if image.shape[-1] == 4:
            image = image[:, :, :-1]
        
        img = np.array(image, dtype = np.uint8)

        # Let this serve as the name for our datas
        img_name = dir[:-4]

        # Since the frame index is inbuilt into the file names, we will leverage that
        frame_idx = int(dir[6:10])

        # Get the coordinates that corresponds to the frame
        # Originally in coords, the dimensions correspond to (human id, frame id, xy coords)
        coords_of_frame = coords[:, frame_idx, :]

        # Problem with this is in coords, the non-existent humans are notated as having the coordinates (0, 0)
        # So we have to filter that off
        # Luckily they have floating point precision in this dataset so it is quite unlikely we purge innocent data points
        c = [coords_of_frame[i, (1, 0)] for i in range(coords_of_frame.shape[0]) if coords_of_frame[i, 0] > 0 and coords_of_frame[i, 1] > 0]

        # Turn it back into a numpy array and perform the int cast
        c = np.array(c, dtype = int)

        if _check_frame:
            check_frame(c, img)

        dmap = coords_to_density_map(c)

        # Make the dictionaries
        imgs[img_name] = (i, img)
        rois[img_name] = (i, roi)
        dmaps[img_name] = (i, dmap)
    
    p.finish()

    # Make the data
    make_data(imgs, rois, dmaps, export)
    

# Entry point
# Test your code first cuz making density maps takes foreve
if __name__ == "__main__":
    # This is convenint if we want to have 20 terminals open at the same time doing our thing
    process = int(sys.argv[1])
    # This is also convenient if we want to discover horribly labelled datasets like the unity one with bugs
    try:
        should_check_frame = bool(sys.argv[2])
    except: 
        should_check_frame = False

    # Can't check image for venice
    if process == 1:
        venice(SOURCE_PATH="./venice/train_data", EXPORT_PATH="./datas/Venice Train")
    if process == 2:
        venice(SOURCE_PATH="./venice/test_data", EXPORT_PATH="./datas/Venice Test")

    # Unity test batches
    if process == 3:
        unity(SOURCE_PATH="./unity/test batch 1", EXPORT_PATH="./datas/Unity Test Batch 1", _check_frame = should_check_frame)
    
    # 21 - 29 corresponds to unity batch 1-9. Idk how many batchs I will make
    if process > 20 and process < 30:
        batch_idx = str(process - 20)
        unity(SOURCE_PATH="./unity/Batch " + batch_idx, EXPORT_PATH="./datas/Unity Batch " + batch_idx, _check_frame = should_check_frame)
    
    # 11 - 19 corresponds to crowd flow 1-9 (if there is ever 6-9)
    # Crowd flow is mostly a lost cause
    if process > 10 and process < 20:
        im_number = str(process - 10)
        crowdflow("./TUBCrowdFlow", export="./datas/CrowdFlow IM0" + im_number, ds_name="IM0" + im_number, _check_frame = should_check_frame)
    