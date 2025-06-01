"""

Description: This file houses code to train our models.

2025-05-23

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from kmeans_pytorch import kmeans

# LOAD DATASET
from duster import Duster, Images

import argparse
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_Q(centroids, latents):

    numerator = (1 + torch.linalg.vector_norm(latents.unsqueeze(1) - centroids.unsqueeze(0).float(), dim = -1))**(-1)

    hollow_matrix = torch.ones_like(numerator)
    diag_inds = list(range(hollow_matrix.shape[0]))
    hollow_matrix[diag_inds, diag_inds] = 0

    denominator = numerator @ hollow_matrix

    return numerator / denominator

def get_P(Q):
    f = Q.sum(axis = 0)

    numerator = Q**2 / f.unsqueeze(0)

    hollow_matrix = torch.ones_like(numerator)
    diag_inds = list(range(hollow_matrix.shape[0]))
    hollow_matrix[diag_inds, diag_inds] = 0

    denominator = numerator @ hollow_matrix

    return numerator / denominator

def KL(P, Q):
    return (P * torch.log(P / Q)).sum(axis = 1)

def initialize_centroids(model, dataloader, k):
    
    latents = []

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)

        latents.extend(
            model(images)
        )

    cluster_ids, centroids = kmeans(
        X = latents, num_clusters = k, distance = "euclidean", device = device
    )

    return centroids

def train():

    epochs = 100
    lr = 1.0e-3
    batch_size = 256
    k = ?

    train_data = Images("../data/processed/", split = "train") # LOAD DATASET USING CHARIS's CODE
    valid_data = Images("../data/processed/", split = "valid") # ^

    train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = True)
    valid_dataloader = DataLoader(valid_data, batch_size = batch_size, shuffle = True)

    DINO = torch.hub.load("facebookresearch/dino:main", "dino_vits16") # DINO
    DINO = DINO.to(device)

    centroids = initialize_centroids(DINO, train_dataloader, k)

    model = Duster(feature_extractor = DINO, centroids = centroids)

    optimizer = torch.optim.AdamW(model.parameters(), lr = lr)


    for e in range(epochs):

        model.train()

        for images, labels in train_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            centroids, latents = model(images)

            Q = get_Q(centroids, latents)
            P = get_P(Q)
            loss = KL(P, Q)

            loss.backward()
            optimizer.step()

        model.eval()

        for images, labels in valid_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            centroids, latents = model(images)




if __name__ == "__main__":
    train()