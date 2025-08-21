import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import time
import random

from backbone import EmbedNetwork
from loss import TripletLoss
from triplet_selector import BatchHardTripletSelector
from batch_sampler import BatchSampler
from datasets.animal_data_set import AnimalDataSet
from optimizer import AdamOptimWrapper
from logger import logger
from embed import embed
from eval import evaluate

def train():
    ## setup
    torch.multiprocessing.set_sharing_strategy('file_system')
    if not os.path.exists('./res'): os.makedirs('./res')

    ## model and loss
    logger.info('setting up backbone model and loss')
    model = EmbedNetwork().cuda()
    model = nn.DataParallel(model)
    triplet_loss = TripletLoss(margin = None).cuda() # no margin means soft-margin

    ## optimizer
    logger.info('creating optimizer')
    optim = AdamOptimWrapper(model.parameters(), lr = 1e-4, wd = 0, t0 = 15000, t1 = 25000)

    ## dataloader
    selector = BatchHardTripletSelector()
    ds = AnimalDataSet('datasets/animal-clef-2025/metadata.csv', is_train = True)
    sampler = BatchSampler(ds, 18, 4)
    dl = DataLoader(ds, batch_sampler = sampler, num_workers = 4)
    diter = iter(dl)

    ## train
    logger.info('start training ...')
    loss_avg = []
    count = 0
    epochs = 0
    t_start = time.time()
    while True:
        try:
            imgs, lbs = next(diter)
        except StopIteration:
            diter = iter(dl)
            imgs, lbs = next(diter)
            epochs += 1

        model.train()
        imgs = imgs.cuda()
        lbs = lbs.cuda()
        embds = model(imgs)
        anchor, positives, negatives = selector(embds, lbs)

        loss = triplet_loss(anchor, positives, negatives)
        optim.zero_grad()
        loss.backward()
        optim.step()

        loss_avg.append(loss.detach().cpu().numpy())
        if count % 20 == 0 and count != 0:
            loss_avg = sum(loss_avg) / len(loss_avg)

            #Do Eval Here:
            if count % 1000 == 0 and count !=0:
                total = len(ds.imgs)
                indices = list(range(total))
                selected_indices = random.sample(indices, total // 2)

                # Select matching images and labels using those indices
                selected_query = [ds.imgs[i] for i in selected_indices]
                selected_query_labels = [ds.label_imgs[i] for i in selected_indices]

                embedded_dataset = embed(model, ds.imgs, augment = False)
                embedded_query = embed(model, selected_query, augment = True)
            
                evaluate(torch.stack(embedded_dataset).squeeze(1).cpu(), torch.tensor(ds.label_imgs).cpu(), torch.stack(embedded_query).squeeze(1).cpu(), torch.tensor(selected_query_labels).cpu())
                torch.save(model.module.state_dict(), f"./res/model_lynx_{epochs}.pkl")

            t_end = time.time()
            time_interval = t_end - t_start
            logger.info('epoch {}, iter: {}, loss: {:4f}, lr: {:4f}, time: {:3f}'.format(epochs, count, loss_avg, optim.lr, time_interval))
            loss_avg = []
            t_start = t_end
            

        count += 1
        if count == 25000: break

    ## dump model
    logger.info('saving trained model')
    torch.save(model.module.state_dict(), './res/model.pkl')

    logger.info('everything finished')

if __name__ == '__main__':
    train()
