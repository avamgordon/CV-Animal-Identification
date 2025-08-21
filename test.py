#!/usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

import sys
import os
import logging
import time
import itertools
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
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import torch.nn.functional as F
import random

TURTLE_THRESHOLD = 10
LYNX_THRESHOLD = 8
SAL_THRESHOLD = 10
PREDICT_TEST_DATA = False
CALCULATE_THRESHOLD = True

def test():

    transform = transforms.Compose([ 
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    # Augmentation Transformations
    augment_transform = transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
     ])

    logger.info('restoring model')

    #Classifier
    model_classifier = EmbedNetwork()
    model_classifier.load_state_dict(torch.load('./model_737.pkl', map_location=torch.device('cpu')))
    model_classifier = nn.DataParallel(model_classifier)
    model_classifier.eval()

    #Turtle
    model_turtle = EmbedNetwork()
    model_turtle.load_state_dict(torch.load('./model_turtle_875.pkl', map_location=torch.device('cpu')))
    model_turtle = nn.DataParallel(model_turtle)
    model_turtle.eval()

    #Lynx
    model_lynx = EmbedNetwork()
    model_lynx.load_state_dict(torch.load('./model_lynx_2250.pkl', map_location=torch.device('cpu')))
    model_lynx = nn.DataParallel(model_lynx)
    model_lynx.eval()

    #Sala
    model_sala = EmbedNetwork()
    model_sala.load_state_dict(torch.load('./model_sala_125.pkl', map_location=torch.device('cpu')))
    model_sala = nn.DataParallel(model_sala)
    model_sala.eval()

    datapath = 'datasets/animal-clef-2025/metadata.csv'
    metadata = pd.read_csv(datapath)

    #This will simulate "new" queries to the DB and see how far away they are from existing queries
    #It will print results to terminal
    if(CALCULATE_THRESHOLD):
        
        #Build unseen test data set
        unique_labels = set()
        for _, row in metadata.iterrows():
            identity = row['identity']
            unique_labels.add(identity)
        unseen_test_labels = random.sample(list(unique_labels), int(0.2 * len(unique_labels)))
        
        stored_feature_vectors_pre_embedded = []
        unseen_feature_vectors = []
        #Store all the encoded vectors
        for _, row in metadata.iterrows():
            identity = row['identity']
            path = row['path']
            # Skip if identity is NaN (test query) or part of test data
            if pd.notna(identity):
                if identity in unseen_test_labels:
                    unseen_feature_vectors.append((identity, path))
                else:
                    stored_feature_vectors_pre_embedded.append((identity, path))
        
        # Compute thresholds
        total = len(stored_feature_vectors_pre_embedded)
        indices = list(range(total))
        selected_indices = random.sample(indices, total // 2)
        selected_validation = [stored_feature_vectors_pre_embedded[i] for i in selected_indices]

        stored_feature_vectors = []
        stored_turtle_vectors = []
        stored_lynx_vectors = []
        stored_sala_vectors = []
        for identity, path in stored_feature_vectors_pre_embedded:
            image = Image.open('datasets/animal-clef-2025/'+ path)
            image = transform(image).unsqueeze(0)
            with torch.no_grad():

                if "Turtle" in identity:
                    turtle_vector = model_turtle(image)
                    stored_turtle_vectors.append((identity, turtle_vector))
                elif "Lynx" in identity:
                    lynx_vector = model_lynx(image)
                    stored_lynx_vectors.append((identity, lynx_vector))
                elif "Sala" in identity:
                    sala_vector = model_sala(image)
                    stored_sala_vectors.append((identity, sala_vector))

                vector = model_classifier(image)
                stored_feature_vectors.append((identity, vector))

        turtle_similarity_seen = 0
        turtles_seen = 0
        lynx_similarity_seen = 0
        lynx_seen = 0
        sal_similarity_seen = 0
        sal_seen = 0
        for identity, path in selected_validation:
            image = Image.open('datasets/animal-clef-2025/'+ path)
            image = augment_transform(image).unsqueeze(0)
            with torch.no_grad(): 
                query_vector = model_classifier(image)
                best_match, similarity = find_closest_match(stored_feature_vectors, query_vector)
                if "Turtle" in best_match:

                    turtle_vector = model_turtle(image)
                    best_match_turtle, similarity_turtle = find_closest_match(stored_turtle_vectors, turtle_vector)
                    turtle_similarity_seen += similarity_turtle
                    turtles_seen += 1

                elif "Lynx" in best_match:

                    lynx_vector = model_lynx(image)
                    best_match_lynx, similarity_lynx = find_closest_match(stored_lynx_vectors, lynx_vector)
                    lynx_similarity_seen += similarity_lynx
                    lynx_seen += 1

                elif "Sala" in best_match:

                    sala_vector = model_sala(image)
                    best_match_sala, similarity_sala = find_closest_match(stored_sala_vectors, sala_vector)
                    sal_similarity_seen += similarity_sala
                    sal_seen += 1
        
        turtle_similarity_unseen = 0
        turtles_unseen = 0
        lynx_similarity_unseen = 0
        lynx_unseen = 0
        sal_similarity_unseen = 0
        sal_unseen = 0
        for identity, path in unseen_feature_vectors:
            image = Image.open('datasets/animal-clef-2025/'+ path)
            image = transform(image).unsqueeze(0)
            with torch.no_grad(): 
                query_vector = model_classifier(image)
                best_match, similarity = find_closest_match(stored_feature_vectors, query_vector)
                if "Turtle" in best_match:
                    turtle_vector = model_turtle(image)
                    best_match_turtle, similarity_turtle = find_closest_match(stored_turtle_vectors, turtle_vector)
                    turtle_similarity_unseen += similarity_turtle
                    turtles_unseen += 1
                elif "Lynx" in best_match:
                    lynx_vector = model_lynx(image)
                    best_match_lynx, similarity_lynx = find_closest_match(stored_lynx_vectors, lynx_vector)
                    lynx_similarity_unseen += similarity_lynx
                    lynx_unseen += 1
                elif "Sala" in best_match:
                    sala_vector = model_sala(image)
                    best_match_sala, similarity_sala = find_closest_match(stored_sala_vectors, sala_vector)
                    sal_similarity_unseen += similarity_sala
                    sal_unseen += 1

        turtle_threshold = ((turtle_similarity_unseen / turtles_unseen) - (turtle_similarity_seen / turtles_seen)) / 2 + (turtle_similarity_seen / turtles_seen)
        lynx_threshold = ((lynx_similarity_unseen / lynx_unseen) - (lynx_similarity_seen / lynx_seen)) / 2 + (lynx_similarity_seen / lynx_seen)
        sal_threshold = ((sal_similarity_unseen / sal_unseen) - (sal_similarity_seen / sal_seen)) / 2 + (sal_similarity_seen / sal_seen)

        print("--------")
        print(f"Seen Turtle Similarity: {turtle_similarity_seen / turtles_seen}")
        print(f"Unseen Turtle Similarity: {turtle_similarity_unseen / turtles_unseen}")
        print(f"Turtle Threshold: {turtle_threshold}")
        print("--------")
        print(f"Seen Lynx Similarity: {lynx_similarity_seen / lynx_seen}")
        print(f"Unseen Lynx Similarity: {lynx_similarity_unseen / lynx_unseen}")
        print(f"Lynx Threshold: {lynx_threshold}")
        print("--------")
        print(f"Seen Sal Similarity: {sal_similarity_seen / sal_seen}")
        print(f"Unseen Sal Similarity: {sal_similarity_unseen / sal_unseen}")
        print(f"Sal Threshold: {sal_threshold}")
        print("--------")

    #This will predict the values for the query datasets and write them out to a submission file to submit on Kaggle
    if(PREDICT_TEST_DATA):
        #Store all the encoded vectors
        stored_feature_vectors = []
        stored_turtle_vectors = []
        stored_lynx_vectors = []
        stored_sala_vectors = []
        for _, row in metadata.iterrows():
            identity = row['identity']
            path = row['path']

            if pd.notna(identity): 
                image = Image.open('datasets/animal-clef-2025/'+ path)
                image = transform(image).unsqueeze(0)
                with torch.no_grad(): 
                    feature_vector = model_classifier(image)
                    stored_feature_vectors.append((identity, feature_vector))

                    if "Turtle" in identity:
                        turtle_vector = model_turtle(image)
                        stored_turtle_vectors.append((identity, turtle_vector))
                    elif "Lynx" in identity:
                        lynx_vector = model_lynx(image)
                        stored_lynx_vectors.append((identity, lynx_vector))
                    elif "Sala" in identity:
                        sala_vector = model_sala(image)
                        stored_sala_vectors.append((identity, sala_vector))

        with open("submission_8.csv", "w") as file:
            file.write("image_id,identity\n")  # Example header row

            for _, row in metadata.iterrows():
                identity = row['identity']
                path = row['path']
                if pd.isna(identity): 
                    image = Image.open('datasets/animal-clef-2025/'+ path)
                    image = transform(image).unsqueeze(0)
                    with torch.no_grad(): 
                        query_vector = model_classifier(image)
                        best_match, _ = find_closest_match(stored_feature_vectors, query_vector)
                                 
                        if "Turtle" in best_match:

                            turtle_vector = model_turtle(image)
                            best_match_turtle, similarity_turtle = find_closest_match(stored_turtle_vectors, turtle_vector)

                            if similarity_turtle < TURTLE_THRESHOLD:
                                file.write(f"{row['image_id']},{best_match_turtle}\n")
                                print(best_match_turtle)
                            else:
                                file.write(f"{row['image_id']},new_individual\n")
                                print("new_individual")

                        elif "Lynx" in best_match:

                            lynx_vector = model_lynx(image)
                            best_match_lynx, similarity_lynx = find_closest_match(stored_lynx_vectors, lynx_vector)

                            if similarity_lynx < LYNX_THRESHOLD:
                                file.write(f"{row['image_id']},{best_match_lynx}\n")
                                print(best_match_lynx)
                            else:
                                file.write(f"{row['image_id']},new_individual\n")
                                print("new_individual")

                        elif "Sala" in best_match:
                            sala_vector = model_sala(image)
                            best_match_sala, similarity_sala = find_closest_match(stored_sala_vectors, sala_vector)

                            if similarity_sala < SAL_THRESHOLD:
                                file.write(f"{row['image_id']},{best_match_sala}\n")
                                print(best_match_sala)
                            else:
                                file.write(f"{row['image_id']},new_individual\n")
                                print("new_individual")

                        else:
                            file.write(f"{row['image_id']},new_individual\n")
                            print("new_individual")

def find_closest_match(feature_vectors, query_vector):

    min_distance = float('inf')
    closest_match = None

    if not feature_vectors:
        return None, float('inf')

    query_vector = query_vector.float().cpu().unsqueeze(0)

    for identity, stored_vector in feature_vectors:
        stored_vector = stored_vector.float().cpu().unsqueeze(0)

        distance = F.pairwise_distance(query_vector, stored_vector).item()

        if distance < min_distance:
            min_distance = distance
            closest_match = identity

    return closest_match, min_distance

if __name__ == '__main__':
    test()
