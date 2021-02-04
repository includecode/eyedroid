#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 23:58:38 2020

@author: pavel
"""
import NetworkConf
import torch
import torch.optim as optim
import torch.nn.functional as F

 
class Train_multiple_epochs():
    def train(self, train_set):
        network = NetworkConf.Network()
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
        optimizer = optim.Adam(network.parameters(), lr=0.01)
         
        def get_num_correct(preds, labels):
            return preds.argmax(dim=1).eq(labels).sum().item()
        
        for epoch in range(10):
            
            total_loss = 0
            total_correct = 0
            
            for batch in train_loader: # Get Batch
                images, labels = batch 
         
                preds = network(images) # Pass Batch
                loss = F.cross_entropy(preds, labels) # Calculate Loss
         
                optimizer.zero_grad()
                loss.backward() # Calculate Gradients
                optimizer.step() # Update Weights
         
                total_loss += loss.item()
                total_correct += get_num_correct(preds, labels)
         
            print(
                "epoch", epoch, 
                "total_correct:", total_correct, 
                "loss:", total_loss
            )