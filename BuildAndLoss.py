#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 22:52:56 2020

@author: pavel
"""
import torch.nn.functional as F
import torch.optim as optim

class BuildAndLossClass(object):
    def build_and_compute_loss(self, network, images, labels):
        preds = network(images)
        loss = F.cross_entropy(preds, labels) # Calculating the loss
         
        loss.item()
         
        def get_num_correct(preds, labels):
            return preds.argmax(dim=1).eq(labels).sum().item()
         
        get_num_correct(preds, labels)
        #Calculating the Gradients
        network.conv1.weight.grad
         
        loss.backward() 
         
        network.conv1.weight.grad.shape
         
        #Updating the Weights
        optimizer = optim.Adam(network.parameters(), lr=0.01)
        optimizer.step() # Updating the weights
         
        preds = network(images)
        loss.item()