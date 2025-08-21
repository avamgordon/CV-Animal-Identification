#!/usr/bin/python
# -*- encoding: utf-8 -*-


import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import RandomErasing
from torch.utils.data import Dataset

import numpy as np
from PIL import Image
import pandas as pd
from sklearn import preprocessing
import random

class AnimalDataSet(Dataset):
    '''
    a wrapper of Market1501 dataset
    '''
    def __init__(self, data_path, is_train = True, *args, **kwargs):
        super(AnimalDataSet, self).__init__(*args, **kwargs)

        self.is_train = is_train
        self.data_path = data_path

        metadata = pd.read_csv(data_path)
        self.imgs = []
        self.label_imgs=[]
        self.unseen_test_imgs=[]
        
        '''
        #Build unseen test data set
        unique_labels = set()
        for _, row in metadata.iterrows():
            identity = row['identity']
            unique_labels.add(identity)
        unseen_test_labels = random.sample(list(unique_labels), int(0.2 * len(unique_labels)))
        '''
        
        #Build rest of dataset
        for _, row in metadata.iterrows():
            identity = row['identity']
            path = row['path']

            # Skip if identity is NaN (test query) or part of test data
            if pd.notna(identity) and ("Lynx" in identity):
                '''
                if identity in unseen_test_labels:
                    self.unseen_test_imgs.append(path)
                else:
                '''    
                self.imgs.append(path)
                self.label_imgs.append(identity)
        
        all_labels = self.label_imgs  
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(all_labels)
        encoded_labels = label_encoder.transform(all_labels)
        self.label_imgs = encoded_labels

        if is_train:
            self.trans = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
            ])
        else:
            self.trans_tuple = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.486, 0.459, 0.408), (0.229, 0.224, 0.225))
                ])
            self.Lambda = transforms.Lambda(
                lambda crops: [self.trans_tuple(crop) for crop in crops])
            self.trans = transforms.Compose([
                transforms.Resize([224, 224]),
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
                self.Lambda,
            ])

        # Make this for the data sampler to ensure good pairs
        self.lb_img_dict = dict()
        self.label_imgs_uniq = set(self.label_imgs)
        lb_array = np.array(self.label_imgs)
        for lb in self.label_imgs_uniq:
            idx = np.where(lb_array == lb)[0]
            self.lb_img_dict.update({lb: idx})
        
        print(self.lb_img_dict)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open('datasets/animal-clef-2025/' + self.imgs[idx])
        img = self.trans(img)
        return img, self.label_imgs[idx]
