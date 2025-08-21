import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

import pickle
import numpy as np
import sys
import logging
from PIL import Image
import torchvision.transforms as transforms


def embed(model, data, augment = True):
    ## logging
    FORMAT = '%(levelname)s %(filename)s:%(lineno)d: %(message)s'
    logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
    logger = logging.getLogger(__name__)

    # Transformations
    transform = transforms.Compose([ 
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    augment_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
     ])
    
    model.eval()
    torch.cuda.empty_cache()

    ## embedding samples
    logger.info('start embedding')

    embedded_vectors = []
    for image_path in data:

        img = Image.open('datasets/animal-clef-2025/' + image_path)
        img_tensor = None

        if augment_transform:
            img_tensor = augment_transform(img)
        else:
            img_tensor = transform(img)

        img_tensor = img_tensor.cuda()
        with torch.no_grad():
            embedding = model(img_tensor.unsqueeze(0))
            embedded_vectors.append(embedding)

    logger.info('embedding finished')
    
    return embedded_vectors
