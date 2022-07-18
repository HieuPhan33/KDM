# -*- coding: utf-8 -*-
"""
Created on 20/08/2020 12:51 pm

@author: Soan Duong, UOW
"""
# Standard library imports
# Third party imports
import torch.nn as nn
import torch
import numpy as np
import sklearn

# Local application imports
from . import base
from . import functional as F
from .base import Activation
import torch.nn.functional as func

class IoU(base.Metric):
    __name__ = 'iou_score'

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    # def forward(self, y_pr, y_gt):
    #     y_pr = self.activation(y_pr)
    #     return F.iou(y_pr, y_gt,
    #                  eps=self.eps,
    #                  threshold=self.threshold,
    #                  ignore_channels=self.ignore_channels)
    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.iou(y_pr, y_gt)


#     y_pr = self.activation(y_pr)
#     return F.iou(y_pr, y_gt,
#                  eps=self.eps,
#                  threshold=self.threshold,
#                  ignore_channels=self.ignore_channels)

class MacroIoU(base.Metric):
    __name__ = 'macro_iou_score'

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        pred = torch.argmax(y_pr, dim=1)
        return sklearn.metrics.jaccard_score(y_gt.flatten().detach().cpu().numpy(),pred.flatten().detach().cpu().numpy(),
                                             average='macro')

# class MacroIoU(base.Metric):
#     __name__ = 'macro_iou_score'
#
#     def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
#         super().__init__(**kwargs)
#         self.eps = eps
#         self.threshold = threshold
#         self.activation = Activation(activation)
#         self.ignore_channels = ignore_channels
#
#     def forward(self, y_pr, y_gt):
#         y_pr = self.activation(y_pr)
#         return F.macro_iou(y_pr, y_gt,
#                      eps=self.eps,
#                      threshold=self.threshold,
#                      ignore_channels=self.ignore_channels)

def iou(pred,target):
    pred = torch.argmax(pred,dim=0)
    return sklearn.metrics.jaccard_score(pred.flatten().detach().cpu().numpy(),
                                  target.flatten().detach().cpu().numpy(),
                                  average='macro')
class Fscore(base.Metric):
    __name__ = 'f1_score'

    def __init__(self, beta=1, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.f_score(y_pr, y_gt,
                         eps=self.eps,
                         beta=self.beta,
                         threshold=self.threshold,
                         ignore_channels=self.ignore_channels)


class Accuracy(base.Metric):
    __name__ = 'accuracy_score'

    def __init__(self, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.accuracy(y_pr, y_gt,
                          threshold=self.threshold,
                          ignore_channels=self.ignore_channels)


class Recall(base.Metric):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.recall(y_pr, y_gt,
                        eps=self.eps,
                        threshold=self.threshold,
                        ignore_channels=self.ignore_channels)


class Precision(base.Metric):

    def __init__(self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.precision(y_pr, y_gt,
                           eps=self.eps,
                           threshold=self.threshold,
                           ignore_channels=self.ignore_channels)

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result

class Dice(base.Metric):
    __name__ = 'iou_score'

    def __init__(self, smooth=1.0, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.smooth = 1.0
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        y_pr = torch.argmax(y_pr, dim=1)
        return F.dice(y_pr, y_gt)

# def dice_coeff(pred, target):
#     smooth = 1.
#     num = pred.size(0)
#     m1 = pred.view(num, -1).float()  # Flatten
#     m2 = target.view(num, -1).float()  # Flatten
#     intersection = (m1 * m2).sum().float()
#
#     return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def accuracy(pred,target):
    pred = torch.argmax(pred, dim=1)
    acc = sklearn.metrics.accuracy_score(target.flatten().detach().cpu().numpy(),
                                         pred.flatten().detach().cpu().numpy())
    return acc

def average_accuracy(pred,target):
    pred = torch.argmax(pred, dim=1)
    aa = sklearn.metrics.recall_score(target.flatten().detach().cpu().numpy(),
                                         pred.flatten().detach().cpu().numpy(),
                                      average='macro')
    return aa

def kappa(pred,target):
    pred = torch.argmax(pred, dim=1)
    kappa = sklearn.metrics.cohen_kappa_score(target.flatten().detach().cpu().numpy(),
                                         pred.flatten().detach().cpu().numpy(),labels=[0,1,2,3,4])
    return kappa



def dice_coeff(pred,target):
    pred = torch.argmax(pred, dim=1)
    dice = sklearn.metrics.f1_score(target.flatten().detach().cpu().numpy(),
                                    pred.flatten().detach().cpu().numpy(),
                                        average='macro')
    return dice

class BinaryDiceLoss(nn.Module):
    """Dice loss of binary class
    Args:
        smooth: A float number to smooth loss, and avoid NaN error, default: 1
        p: Denominator value: \sum{x^p} + \sum{y^p}, default: 2
        predict: A tensor of shape [N, *]
        target: A tensor of shape same with predict
        reduction: Reduction method to apply, return mean over batch if 'mean',
            return sum if 'sum', return a tensor of shape [N,] if 'none'
    Returns:
        Loss tensor according to arg reduction
    Raise:
        Exception if unexpected reduction
    """
    def __init__(self, smooth=1, p=2, reduction='mean'):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = smooth
        self.p = p
        self.reduction = reduction

    def forward(self, predict, target):
        assert predict.shape[0] == target.shape[0], "predict & target batch size don't match"
        predict = predict.contiguous().view(predict.shape[0], -1)
        target = target.contiguous().view(target.shape[0], -1)

        num = torch.sum(torch.mul(predict, target), dim=1) + self.smooth
        den = torch.sum(predict.pow(self.p) + target.pow(self.p), dim=1) + self.smooth

        loss = 1 - num / den

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        elif self.reduction == 'none':
            return loss
        else:
            raise Exception('Unexpected reduction {}'.format(self.reduction))


class DiceLoss(nn.Module):
    """Dice loss, need one hot encode input
    Args:
        weight: An array of shape [num_classes,]
        ignore_index: class index to ignore
        predict: A tensor of shape [N, C, *]
        target: A tensor of same shape with predict
        other args pass to BinaryDiceLoss
    Return:
        same as BinaryDiceLoss
    """
    def __init__(self, weight=None, ignore_index=None, **kwargs):
        super(DiceLoss, self).__init__()
        self.kwargs = kwargs
        self.weight = weight
        self.ignore_index = ignore_index

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        dice = BinaryDiceLoss(**self.kwargs)
        total_loss = 0
        predict = func.softmax(predict, dim=1)

        for i in range(target.shape[1]):
            if i != self.ignore_index:
                dice_loss = dice(predict[:, i], target[:, i])
                if self.weight is not None:
                    assert self.weight.shape[0] == target.shape[1], \
                        'Expect weight shape [{}], get[{}]'.format(target.shape[1], self.weight.shape[0])
                    dice_loss *= self.weights[i]
                total_loss += dice_loss

        return total_loss/target.shape[1]