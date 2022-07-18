# -*- coding: utf-8 -*-
"""
Created on 25/08/2020 6:42 pm

@author: Soan Duong, UOW
"""
# Standard library imports
import numpy as np

# Third party imports
import torch
import torch.nn as nn

# Local application imports

# 2D loss example (used, for example, with image inputs)
N, C = 40, 5
loss = nn.CrossEntropyLoss()
# loss = nn.NLLLoss()
# input is of size N x C x height x width
data = torch.randn(N, C, 217, 409).cuda()
m = nn.LogSoftmax(dim=1)
# each element in target has to have 0 <= value < C
target = torch.empty(N, 217, 409, dtype=torch.long).random_(0, C).cuda()
# y_pr = m(data)
y_pr = data
output = loss(y_pr, target)

