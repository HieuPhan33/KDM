# -*- coding: utf-8 -*-
"""
Created on 23/11/2020 1:36 pm

@author: Soan Duong, Hieu Phan UOW
"""
# Standard library imports
# Third party imports
import torch
import torch.nn as nn

# Local application imports
from . import base
from . import functional as F
import torch.nn.functional as func
from .base import Activation


class JaccardLoss(base.Loss):
    def __init__(self, eps=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)

        self.eps = eps
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_gt, y_pr):
        y_pr = self.activation(y_pr)
        return 1 - F.jaccard(y_pr, y_gt,
                             eps=self.eps,
                             threshold=None,
                             ignore_channels=self.ignore_channels)


class DiceLoss(base.Loss):
    def __init__(self, eps=1., beta=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)

        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_gt, y_pr):
        y_pr = self.activation(y_pr)
        return 1 - F.f_score(y_pr, y_gt,
                             beta=self.beta,
                             eps=self.eps,
                             threshold=None,
                             ignore_channels=self.ignore_channels)


class FocalLoss(nn.Module):

    def __init__(self, weight=None,
                 gamma=2., reduction='none'):
        nn.Module.__init__(self)
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input_tensor, target_tensor):
        log_prob = nn.functional.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob) ** self.gamma) * log_prob,
            target_tensor,
            weight=self.weight,
            reduction=self.reduction
        )


# Cross-entropy Loss
class FSCELoss(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(FSCELoss, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label)

    def forward(self, inputs, *targets, weights=None, **kwargs):
        loss = 0.0
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            if weights is None:
                weights = [1.0] * len(inputs)

            for i in range(len(inputs)):
                if len(targets) > 1:
                    target = self._scale_target(targets[i], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.criterion(inputs[i], target)
                else:
                    target = self._scale_target(targets[0], (inputs[i].size(2), inputs[i].size(3)))
                    loss += weights[i] * self.criterion(inputs[i], target)

        else:
            target = self._scale_target(targets[0], (inputs.size(2), inputs.size(3)))
            loss = self.criterion(inputs, target)

        return loss

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().unsqueeze(1).float()
        targets = func.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()


class CrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, weight=None):
        super(CrossEntropy, self).__init__()
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label)

    def forward(self, score, target):
        score = score[-1]
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = func.upsample(
                input=score, size=(h, w), mode='bilinear')

        loss = self.criterion(score, target)

        return loss

class KnowledgeDistillationLoss(nn.Module):
    def __init__(self,temperature=3, ignore_label=-1, weight=None, feat_w=0, resp_w=0):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature
        self.feat_w = feat_w
        self.resp_w = resp_w
        self.response_loss = KDPixelWiseCE(temperature=temperature)
        # self.feature_loss = KDObjectAttention(use_ifv=use_ifv)
        self.feature_loss = KDFeat()
        self.classification_loss = FSCELoss(ignore_label=ignore_label,
                                            weight=weight)

    @staticmethod
    def _scale_target(targets_, scaled_size):
        targets = targets_.clone().unsqueeze(1).float()
        targets = func.interpolate(targets, size=scaled_size, mode='nearest')
        return targets.squeeze(1).long()

    def forward(self,student_output, teacher_output, student_feat, teacher_feat, output):
        teacher_output = func.interpolate(teacher_output, size=(student_output.size(2), student_output.size(3)), mode='nearest')
        output = self._scale_target(output, (student_output.size(2), student_output.size(3)))
        if student_output[0].size() != teacher_output[0].size():
            teacher_output = func.interpolate(teacher_output, (student_output.size(-2), student_output.size(-1)))
        loss = self.resp_w * self.response_loss(student_output,teacher_output)

        if student_feat[0].size() != teacher_feat[0].size():
            teacher_feat = func.interpolate(teacher_feat, (student_feat.size(-2), student_feat.size(-1)))
        loss += self.feat_w * self.feature_loss(torch.sum(student_feat,dim=1).unsqueeze(1), torch.sum(teacher_feat,dim=1).unsqueeze(1))
        loss += self.classification_loss(student_output,output)
        return loss

class OhemCrossEntropy(nn.Module):
    def __init__(self, ignore_label=-1, thres=0.7,
                 min_kept=100000, weight=None):
        super(OhemCrossEntropy, self).__init__()
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_label,
                                             reduction='none')

    def forward(self, score, target, **kwargs):
        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = func.upsample(input=score, size=(h, w), mode='bilinear')
        pred = func.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label

        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_label] = 0
        pred = pred.gather(1, tmp_target.unsqueeze(1))
        pred, ind = pred.contiguous().view(-1, )[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)]
        threshold = max(min_value, self.thresh)

        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold]
        return pixel_losses.mean()

def SpatialSoftmax(feature):
    feature = feature.view(feature.shape[0], feature.shape[1], -1)
    softmax_attention = func.softmax(feature, dim=-1)
    return softmax_attention



class KDFeat(nn.Module):
    def __init__(self):
        super(KDFeat, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, f_S, f_T):
        """
        x1 - previous encoder step feature map
        x2 - current encoder step feature map
        """
        # G^2_sum
        if f_S.size() != f_T.size():
            f_T = func.upsample(f_T, size=(f_S.shape[2],f_S.shape[3]), mode='bilinear', align_corners=True)
        f_S = torch.sum(f_S * f_S, dim=1, keepdim=True)
        f_S = SpatialSoftmax(f_S)
        f_T = torch.sum(f_T * f_T, dim=1, keepdim=True)
        f_T = SpatialSoftmax(f_T)

        # # G^2_sum
        # if f_S.size() != f_T.size():
        #     f_T = func.upsample(f_T, size=(f_S.shape[2],f_S.shape[3]))

        #     f_S = torch.sum(f_S * f_S, dim=1, keepdim=True)
        #     f_S = SpatialSoftmax(f_S)
        #     f_T = torch.sum(f_T * f_T, dim=1, keepdim=True)
        #     #f_T = func.upsample(f_T, size=(f_S.shape[2],f_S.shape[3]))
        #     #f_T = torch.squeeze(self.at_gen_upsample(x2), dim=1)
        #     f_T = SpatialSoftmax((f_T)
        # else:
        #     f_S = torch.sum(f_S * f_S, dim=1, keepdim=True)
        #     f_S = SpatialSoftmax(f_S)
        #     f_T = torch.sum(f_T * f_T, dim=1, keepdim=True)
        #     f_T = SpatialSoftmax(f_T)
        loss = self.criterion(f_S, f_T)
        return loss

    # def forward(self, f_S, f_T):
    #     #print(f_S.size(), f_T.size())
    #     f_S = func.upsample(f_S,size=(f_T.shape[2],f_T.shape[3]))
    #     assert f_S.shape == f_T.shape, f"Shape of student {f_S.shape} - Shape of teacher {f_T.shape}"
    #     f_T.detach()
    #     return self.criterion(f_S, f_T)


class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())  # OR operation
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

class KDPixelWiseCE(nn.Module):
    def __init__(self,temperature=3):
        super(KDPixelWiseCE, self).__init__()
        self.T = temperature

    def forward(self, preds_S, preds_T):
        preds_T.detach()
        assert preds_S[0].shape == preds_T[0].shape,f'the dim of teacher {preds_T.shape} != and student {preds_S.shape}'
        B,C,W,H = preds_S.shape
        preds_S = preds_S.permute(0, 2, 3, 1).contiguous().view(-1, C)
        preds_T = preds_T.permute(0, 2, 3, 1).contiguous().view(-1, C)
        preds_S = func.log_softmax(preds_S / self.T, dim=1)
        preds_T = func.softmax(preds_T / self.T, dim=1)
        preds_T = preds_T + 10 ** (-7)
        preds_T = torch.autograd.Variable(preds_T.data.cuda(), requires_grad=False)
        loss = self.T * self.T * torch.sum(-preds_T*preds_S)/(B*H*W)
        #loss = self.T * self.T * F.kl_div(preds_S, preds_T, reduction="sum") / (H * W)  # o
        return loss

class KDLoss(nn.Module):
    def __init__(self, temperature=3, ignore_label=-1, weight=None, base_feat_w=0, base_resp_w=0, student_loss_w=1.0):
        super(KDLoss, self).__init__()
        self.base_feat_w = base_feat_w
        self.base_resp_w = base_resp_w
        self.feat_w = 0
        self.resp_w = 0
        self.student_loss_w = student_loss_w
        self.response_loss = KDPixelWiseCE(temperature=temperature)
        # self.feature_loss = KDObjectAttention(use_ifv=use_ifv)
        self.feature_loss = KDFeat()
        self.classification_loss = FSCELoss(ignore_label=ignore_label,
                                            weight=weight)

    def update_kd_loss_params(self, iters, max_iters):
        self.feat_w = (iters / max_iters) * self.base_feat_w
        self.resp_w = (iters / max_iters) * self.base_resp_w

    def forward(self, f_results, o_results, targets, semi=False, **kwargs):
        # for f in f_results:
        #     print(f.shape)
        loss = 0
        num_blocks = len(f_results)
        for r in range(num_blocks):
            if r < len(f_results) - 1:
                f_prev, f_next = f_results[r], f_results[-1]
            else:
                f_prev, f_next = None, None
            o_prev, o_next = o_results[r], o_results[-1]
            loss += self.forward_one_session(f_prev, f_next, o_prev, o_next, targets, semi, **kwargs)
        loss += self.classification_loss(o_results[-1], targets)
        return loss

    def forward_one_session(self, f_prev, f_next, o_prev, o_next, targets, semi, **kwargs):
        o_prev = func.interpolate(o_prev, (o_next.size(-2), o_next.size(-1)))
        # f_prev = F.interpolate(f_prev,(f_next.size(-2), f_next.size(-1)))
        # f_s = F.interpolate(f_s, (f_t.size(-2), f_t.size(-1)))
        if semi:  # Semi-supervised training
            cls_loss = 0
        else:
            # cls_loss = self.classification_loss((o_prev, o_next), targets)
            cls_loss = self.classification_loss(o_prev, targets)
        loss = self.student_loss_w * cls_loss
        if self.resp_w > 0 and o_next is not None and o_prev is not None:
            response_loss = self.resp_w * self.response_loss(o_prev, o_next)
            loss += response_loss
        if self.feat_w > 0 and f_next is not None and f_prev is not None:
            # feature_loss = self.feat_w * self.feature_loss(f_prev,f_next,targets,prob_map=None)
            feature_loss = self.feat_w * self.feature_loss(f_prev, f_next)
            loss += feature_loss
        return loss

class L1Loss(nn.L1Loss, base.Loss):
    pass


class MSELoss(nn.MSELoss, base.Loss):
    pass


class CrossEntropyLoss(nn.CrossEntropyLoss, base.Loss):
    pass


class NLLLoss(nn.NLLLoss, base.Loss):
    pass


class NLLLoss2d(nn.NLLLoss2d, base.Loss):
    pass


class BCELoss(nn.BCELoss, base.Loss):
    pass


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss, base.Loss):
    pass
