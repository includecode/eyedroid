#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 23:14:43 2020

@author: pavel
"""
import NetworkConf
import torch
import torch.optim as optim
import torch.nn.functional as F

class Train_single_batch():
    def train(self, train_set):
        #Train Using a Single Batch
        network = NetworkConf.Network()
         
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
        optimizer = optim.Adam(network.parameters(), lr=0.01)
         
        batch = next(iter(train_loader)) # Get Batch
        images, labels = batch
         
        preds = network(images) # Pass Batch
        loss = F.cross_entropy(preds, labels) # Calculate Loss
         
        loss.backward() # Calculate Gradients
        optimizer.step() # Update Weights
         
        print('loss1:', loss.item())
        preds = network(images)
        loss = F.cross_entropy(preds, labels)
        print('loss2:', loss.item())