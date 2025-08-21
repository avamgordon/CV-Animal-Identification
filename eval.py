import numpy as np

from logger import logger
import argparse
from tqdm import tqdm

from utils import pdist_np as pdist


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
            '--gallery_embs',
            dest = 'gallery_embs',
            type = str,
            default = './res/emb_gallery.pkl',
            help = 'path to embeddings of gallery dataset'
            )
    parse.add_argument(
            '--query_embs',
            dest = 'query_embs',
            type = str,
            default = './res/emb_query.pkl',
            help = 'path to embeddings of query dataset'
            )
    parse.add_argument(
            '--cmc_rank',
            dest = 'cmc_rank',
            type = int,
            default = 1,
            help = 'path to embeddings of query dataset'
            )

    return parse.parse_args()


def evaluate(emb_dataset, dataset_labels, emb_query, query_labels):
    
    ## compute and clean distance matrix
    dist_mtx = pdist(emb_query, emb_dataset)
    n_q, n_g = dist_mtx.shape
    indices = np.argsort(dist_mtx, axis = 1)
    matches = dataset_labels[indices] == query_labels[:, np.newaxis]
    matches = matches.cpu().numpy().astype(np.int32)
    all_aps = []
    all_cmcs = []
    logger.info('starting evaluating ...')
    for qidx in tqdm(range(n_q)):
        qpid = query_labels[qidx]

        order = indices[qidx]
        pid_diff = dataset_labels[order] != qpid
        useful = dataset_labels[order] != -1
        keep = np.logical_and(pid_diff, useful)
        match = matches[qidx][keep]

        if not np.any(match): continue

        cmc = match.cumsum()
        cmc[cmc > 1] = 1
        all_cmcs.append(cmc[:1])

        num_real = match.sum()
        match_cum = match.cumsum()
        match_cum = [el / (1.0 + i) for i, el in enumerate(match_cum)]
        match_cum = np.array(match_cum) * match
        ap = match_cum.sum() / num_real
        all_aps.append(ap)
        
        print(f"{qidx} / {n_q}")

    assert len(all_aps) > 0, "NO QUERY MATCHED"
    mAP = sum(all_aps) / len(all_aps)
    all_cmcs = np.array(all_cmcs, dtype = np.float32)
    cmc = np.mean(all_cmcs, axis = 0)

    logger.info('mAP is: {}, cmc is: {}'.format(mAP, cmc))