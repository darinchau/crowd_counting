import random
import os
from PIL import Image
import numpy as np
import h5py
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from utility import searchFile, hehehaha

class listDataset(Dataset):
    def __init__(self, root, root_dir, shape=None, shuffle=True, transform=None,  train=False, seen=0, direct=False, batch_size=1, num_workers=4, gt_code=1):
        if train:
            root = root
        random.shuffle(root)
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.direct=direct 
        self.gt_code = gt_code
        self.root_dir = root_dir

    def __len__(self):
        return self.nSamples
    
    def __getitem__(self, index):
        assert index <= len(self), 'index range error'         
        img_path = self.lines[index]
        img,target = load_data(img_path, self.root_dir, False,code=self.gt_code)
        img_r = load_data(img_path, self.root_dir, False, direct = True, code=self.gt_code)
        if self.direct:
            return img,target,img_r
        return img,target

def load_data(img_path, root_dir, train = True, direct = False, code = 1):
    transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]) ])

    transform2 = transforms.ToTensor()

    gt_path = img_path.replace('.png','.h5').replace("img1",'target'+code).replace('.jpg','.h5').replace("\\", "/")
    img_path = img_path.replace("\\", "/")

    aug = 0

    for root, dir, filenames in os.walk(img_path):
        # root, dir, filenames looks like ./venice/ablation3\4896_004260.jpg [] ['4896_004140.jpg', '4896_004200.jpg', '4896_004260.jpg']
        
        # Process each files
        for i in range(len(filenames)):
            # Process the first image differently
            if i == 0:
                img =  np.array(Image.open(img_path + "/" + filenames[i]).convert('RGB'))
                if direct is True:
                    return transform2(img)
                img = transform(img)
                image = img
                if aug == 1:
                    image = img.transpose(Image.FLIP_LEFT_RIGHT)
                if aug == 2:
                    crop_size = (int(image.shape[1]/2),int(image.shape[2]/2))
                    if random.randint(0,9)<= 3:
                        dx = int(random.randint(0,1)*image.shape[1]*1./2)
                        dy = int(random.randint(0,1)*image.shape[2]*1./2)
                    else:
                        dx = int(random.random()*image.shape[1]*1./2)
                        dy = int(random.random()*image.shape[2]*1./2)
                    image = image[:,dx:crop_size[0]+dx,dy:crop_size[1]+dy]
                if aug == 0:
                    image = image.unsqueeze(dim = 1)
            else:
                new_img =  np.array(Image.open(img_path + "/" + filenames[i]).convert('RGB'))
                new_img = transform(new_img)
                new_image = new_img
                if aug ==1:
                    new_image = new_image.transpose(Image.FLIP_LEFT_RIGHT)
                if aug == 2:
                    new_image = new_image[:,dx:crop_size[0]+dx,dy:crop_size[1]+dy]
                    new_image = new_image.unsqueeze(dim = 1)
                if aug == 0:
                    new_image = new_image.unsqueeze(dim = 1)
                image = torch.cat([image, new_image], axis = 1)


    # Loads the target (density map files) - first try to load directly from path but also try a search
    try:
        gt_file = h5py.File(gt_path)
    except:
        img_name = gt_path.split("/")[-1]
        candidates = searchFile(root_dir + "venice", img_name)
        if len(candidates) != 1:
            raise AssertionError(f"Found incorrect number of files! Files with name {img_name} found at root path {root_dir}/venice: {len(candidates)}")
        # print("HEHEHAHA")
        # print(*candidates)
        gt_file = h5py.File(candidates[0][0] + "/" + candidates[0][1])
    
    target = np.asarray(gt_file['density'])
    if aug == 1:
        target = np.fliplr(target)
    if aug == 2:
        target = target[dx:crop_size[0]+dx,dy:crop_size[1]+dy]
        
    # Note: Using PIL instead of OpenCV produces an average pixel difference of about 0.002. Insignificant sure but also something noteworthy
    t = Image.fromarray(target)
    size = (int(np.floor(image.shape[3]/8)), int(np.floor(image.shape[2]/8)))
    t = t.resize(size, Image.BICUBIC)
    t = np.array(t, dtype = target.dtype) * 64

    return image, t