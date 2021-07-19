import os
import torch
import random
import pickle

import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from torch.utils.data.dataset import Dataset

from libs.utils import *

class Train_Dataset_SLNet(Dataset):

    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        self.datalist = load_pickle(cfg.pickle_dir + 'detector_train_candidates')
        if mode == 'train':
            self.datalist = self.datalist[0:int(len(self.datalist)*(1-cfg.ratio_val))]
        else:
            self.datalist = self.datalist[-int(len(self.datalist)*cfg.ratio_val):]
        self.transform = transforms.Compose([transforms.Resize((cfg.height, cfg.width), 2),
                                             transforms.ToTensor()])
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)

        self.left_num = int(cfg.batch_size['train_line'] * cfg.ratio_left)
        self.mid_num = int(cfg.batch_size['train_line'] * cfg.ratio_mid)
        self.right_num = int(cfg.batch_size['train_line'] * cfg.ratio_right)
        self.neg_num = int(cfg.batch_size['train_line'] * cfg.ratio_neg)
        self.height = cfg.height
        self.width = cfg.width

    def get_image(self, flip, idx):
        img_name = os.path.join(self.cfg.img_dir, self.datalist[idx]['img_path'])
        img = Image.open(img_name)
        if flip == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        width, height = img.size
        return img, torch.FloatTensor([height, width]), self.datalist[idx]['img_path']

    def one_hot_encoding(self, label):
        data=[]
        data=torch.zeros((label.shape[0], 4))
        data[range(label.shape[0]), label]=1
        return np.array(data)
    
    def make_line_data(self, data, name):
        if data[name]['endpts'].shape[0] > 0:
            return np.concatenate((data[name]['endpts'],
                                   self.one_hot_encoding(data[name]['cls']),
                                   data[name]['offset']), axis=1)
        else:
            return np.empty((0,12))
    
    def make_idx_num(self, left_data, mid_data, right_data, neg_data):
        ans=[0, 0, 0, 0]
        data=[left_data.shape[0],mid_data.shape[0],right_data.shape[0],neg_data.shape[0]]
        idx=[]
        i=0

        if data[0] != 0:
            idx.append(0)
        if data[1] != 0:
            idx.append(1)
        if data[2] != 0:
            idx.append(2)
        if data[3] != 0:
            idx.append(3)

        while i<self.cfg.batch_size['train_line']:
            n=random.choice(idx)
            if ans[n] < data[n]:
                ans[n]+=1
                i+=1

        return ans[0], ans[1], ans[2], ans[3]
    

    def get_data(self, flip, idx):

        data = self.datalist[idx]
        left_data = self.make_line_data(data, 'left_line')
        mid_data = self.make_line_data(data, 'mid_line')
        right_data = self.make_line_data(data, 'right_line')
        neg_data = self.make_line_data(data, 'neg_line')
        # random sampling
        lf_num, md_num, rg_num, ng_num=self.make_idx_num(left_data,mid_data,right_data,neg_data)
        left_idx = torch.randperm(left_data.shape[0])[:lf_num]
        mid_idx = torch.randperm(mid_data.shape[0])[:md_num]
        right_idx = torch.randperm(right_data.shape[0])[:rg_num]
        neg_idx = torch.randperm(neg_data.shape[0])[:ng_num]

        left_data=left_data[left_idx.numpy()]
        mid_data=mid_data[mid_idx.numpy()]
        right_data=right_data[right_idx.numpy()]
        neg_data=neg_data[neg_idx.numpy()]


        if flip == 1:
            left_data[:, 0] = self.width - 1 - left_data[:, 0]
            left_data[:, 2] = self.width - 1 - left_data[:, 2]
            left_data[:,8] = -1 * left_data[:, 8]
            left_data[:, 10] = -1 * left_data[:, 10]

            left_data[:, [5,7]]= left_data[:, [7,5]]


            mid_data[:, 0] = self.width - 1 - mid_data[:, 0]
            mid_data[:, 2] = self.width - 1 - mid_data[:, 2]
            mid_data[:, 8] = -1 * mid_data[:, 8]
            mid_data[:, 10] = -1 * mid_data[:, 10]

            right_data[:, 0] = self.width - 1 - right_data[:, 0]
            right_data[:, 2] = self.width - 1 - right_data[:, 2]
            right_data[:, 8] = -1 * right_data[:, 8]
            right_data[:, 10] = -1 * right_data[:, 10]

            right_data[:, [5,7]]= right_data[:, [7,5]]


            neg_data[:, 0] = self.width - 1 - neg_data[:,0]
            neg_data[:, 2] = self.width - 1 - neg_data[:, 2]
            neg_data[:, 8] = -1 * neg_data[:, 8]
            neg_data[:, 10] = -1 * neg_data[:, 10]

            right_data, mid_data, left_data, neg_data

        return left_data, mid_data, right_data, neg_data

    def normalize_point_coord(self, data):
        data[:, 8] = data[:, 8] / self.cfg.width
        data[:, 9] = data[:, 9] / self.cfg.height
        data[:, 10] = data[:, 10] / self.cfg.width
        data[:, 11] = data[:, 11] / self.cfg.height
        return data

    def __getitem__(self, idx):
        # flip = random.randint(0, 1)
        # ******************************* changed ************************* no flip!
        flip=0
        # get pre-processed images
        img, img_size, img_name = self.get_image(flip, idx)
        img = self.transform(img)

        # get candidate lines
        left_data, mid_data, right_data, neg_data = self.get_data(flip, idx)

        left_data = self.normalize_point_coord(left_data)
        mid_data = self.normalize_point_coord(mid_data)
        right_data = self.normalize_point_coord(right_data)

        train_data = np.concatenate((left_data, mid_data, right_data, neg_data), axis=0)
        
        return {'img_rgb': img,
                'img': self.normalize(img),
                'img_size': img_size,
                'img_name': img_name,
                'flip': flip,
                'train_data': train_data}

    def __len__(self):
        return len(self.datalist)



class SEL_Test_Dataset(Dataset):

    def __init__(self, cfg):

        self.cfg = cfg

        self.datalist = load_pickle(cfg.dataset_dir + 'data/test')
        self.transform = transforms.Compose([transforms.Resize((cfg.height, cfg.width), 2),
                                             transforms.ToTensor()])
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)

        self.height = cfg.height
        self.width = cfg.width

    def get_image(self, idx):
        img_name = os.path.join(self.cfg.img_dir,
                                self.datalist['img_path'][idx])
        img = Image.open(img_name).convert("RGB")

        width, height = img.size
        return img, torch.FloatTensor([height, width]), self.datalist['img_path'][idx]

    def get_gtlines(self, idx):
        #changed for crosswalk
        mul_gt = self.datalist['multiple'][idx]
        pri_gt_idx = self.datalist['primary'][idx]
        
        left_gt = mul_gt[pri_gt_idx == 1]
        middle_gt = mul_gt[pri_gt_idx == 2]
        right_gt = mul_gt[pri_gt_idx == 3]
        return left_gt, middle_gt, right_gt, mul_gt


    def __getitem__(self, idx):

        # get pre-processed images
        img, img_size, img_name = self.get_image(idx)
        img = self.transform(img)

        left_gt, middle_gt, right_gt, mul_gt = self.get_gtlines(idx)

        return {'img_rgb': img,
                'img': self.normalize(img),
                'img_size': img_size,
                'img_name': img_name,
                'left_gt': left_gt,
                'middle_gt': middle_gt,
                'right_gt': right_gt,
                'mul_gt': mul_gt}

    def __len__(self):
        return len(self.datalist['img_path'])

class SEL_Hard_Test_Dataset(Dataset):

    def __init__(self, cfg):
        # setting
        self.cfg = cfg
        self.scale = np.float32([cfg.width, cfg.height, cfg.width, cfg.height])

        # load datalist
        self.datalist = load_pickle(cfg.dataset_dir + 'data/SEL_Hard')

        # image transform
        self.transform = transforms.Compose([transforms.Resize((cfg.height, cfg.width), 2),
                                             transforms.ToTensor()])
        self.normalize = transforms.Normalize(mean=cfg.mean, std=cfg.std)

    def get_image(self, idx):
        img_name = os.path.join(self.cfg.img_dir, self.datalist['img_path'][idx])

        img = Image.open(img_name).convert('RGB')
        width, height = img.size
        return img, torch.FloatTensor([height, width]), self.datalist['img_path'][idx]

    def get_gt_endpoints(self, idx):

        pri_gt = self.datalist['primary'][idx]
        mul_gt = self.datalist['multiple'][idx]

        return pri_gt, mul_gt

    def __getitem__(self, idx):

        # get pre-processed images
        img, img_size, img_name = self.get_image(idx)
        img = self.transform(img)

        pri_gt, mul_gt = self.get_gt_endpoints(idx)

        return {'img_rgb': img,
                'img': self.normalize(img),
                'img_size': img_size,
                'img_name': img_name,
                'pri_gt': pri_gt,
                'mul_gt': mul_gt}

    def __len__(self):
        return len(self.datalist['img_path'])
