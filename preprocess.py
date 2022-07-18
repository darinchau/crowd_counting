import mat4py
import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import scipy
import scipy.spatial as ss
import re
import shutil
import cv2
import json

DST_LABEL_PATH = './Venice/density_map_init'

SOURCE_PATH = './Venice/venice'
IMG_ROI_PATH = './Venice/img_roi'
ROI_PATH = './Venice/density_map_init'

IMAGE_NUM = 3

EXPORT_PATH = '/Venice/ablation' + str(IMAGE_NUM)

def gaussian_filter_density_new(gt, sigma=5):
    sha = (720, 1280)
    density = np.zeros(sha, dtype=np.float32)
    for i in range(len(gt)):
        pt2d = np.zeros(sha, dtype=np.float32)
        try:
            pt2d[gt[i][1],gt[i][0]] = 1.
        except:
            pt2d[gt[i][1]-1,gt[i][0]-1] = 1.
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    return density

def make_density_map():
    list1 = searchFile(SOURCE_PATH,'(.*).mat')
    list1.sort()

    for i in range(len(list1)):
        print('{}/{}'.format(i, len(list1)))
        try:
            data = mat4py.loadmat(os.path.join(list1[i][0], list1[i][1]))
        except:
            data = h5py.File(os.path.join(list1[i][0],list1[i][1]), 'r')
        map = gaussian_filter_density_new(data['annotation'],5)
        path = os.path.join(DST_LABEL_PATH,list1[i][1]).replace('.mat','.h5')
        if not os.path.exists(DST_LABEL_PATH):
                os.makedirs(DST_LABEL_PATH)
        with h5py.File(path, 'w') as hf:
                hf['density'] = map
                hf['roi'] = data['roi']


def mycopyfile(srcfile,dstfile):
    if not os.path.isfile(srcfile):
        print("not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(dstfile)   
        if not os.path.exists(fpath):
            os.makedirs(fpath)               
        shutil.copyfile(srcfile,dstfile)     
        print("copy %s -> %s"%( srcfile,dstfile))


def searchFile(pathname,filename):
    matchedFile =[]
    for root,dirs,files in os.walk(pathname):
        for file in files:
            if re.match(filename,file):
                matchedFile.append((root,file))
    return matchedFile

def process_roi():
    list1 = []
    list1 = searchFile(SOURCE_PATH,'(.*).jpg')

    for index in range(len(list1)):
        roi_path = os.path.join(ROI_PATH, list1[index][1].replace('.jpg', '.h5'))
        data = h5py.File(roi_path, 'r')
        src = os.path.join(list1[index][0],list1[index][1])
        tar = os.path.join(IMG_ROI_PATH, list1[index][1])
        mycopyfile(src, tar)
        img = cv2.imread(tar)
        for i in range(3):
            img[:,:,i] = img[:,:,i] * data['roi']
        cv2.imwrite(tar,img)
    
def normal(ind1, ind2, list1):
    if ind1 < 0:
        return False
    if list1[ind1].split('_')[0] != list1[ind2].split('_')[0]:
        return False
    i1 = int(list1[ind1].split('.')[0].split('_')[1])
    i2 = int(list1[ind2].split('.')[0].split('_')[1])
    if (i2-i1) > 60*(ind2-ind1+1):
        print('{} lost more than one frame to {}'.format(list1[ind1], list1[ind2]))
        return False
    return True

def make_3d_dataset():
    train_json = []
    test_json = []

    list1 = []
    list1 = searchFile(IMG_ROI_PATH,'(.*).jpg')
    list1.sort()

    for index in range(len(list1)):
        for place in range(IMAGE_NUM-1,-1,-1):
            if normal(index-place, index, list1):
                mycopyfile(os.path.join(IMG_ROI_PATH, list1[index-place]), os.path.join(EXPORT_PATH, list1[index], list1[index-place]))
            else:
                mycopyfile(os.path.join(IMG_ROI_PATH, list1[index]), os.path.join(EXPORT_PATH, list1[index], list1[index].split('.')[0]+str(place)+'.jpg'))
        if list1[index].split('_')[0] == '4896':
            train_json.append(os.path.join(EXPORT_PATH, list1[index]))
        else:
            test_json.append(os.path.join(EXPORT_PATH, list1[index]))
                
            

    with open('./jsons/train' + str(IMAGE_NUM) + '.json', 'w', encoding='utf-8') as json_file:
        json.dump(train_json, json_file, indent=1)
        print(len(train_json))

    with open('./jsons/test' + str(IMAGE_NUM) + '.json', 'w', encoding='utf-8') as json_file:
        json.dump(test_json, json_file, indent=1)
        print(len(test_json))

if __name__ == "__main__":
    make_density_map()
