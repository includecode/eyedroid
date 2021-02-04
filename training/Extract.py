#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  5 22:27:24 2020

@author: pavel
"""

import torchvision
import torchvision.transforms as transforms

class ExtractData(object):
    def extract_data(self):
        train_set = torchvision.datasets.FashionMNIST(
            root='./data/FashionMNIST'
            ,train=True
            ,download=True
            ,transform=transforms.Compose([
                transforms.ToTensor()
            ])
        )
        return train_set