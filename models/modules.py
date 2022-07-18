"""
Created on 23/11/2020 1:40 pm

@author: Hieu Phan, UOW
"""

import os
import pdb
import math
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.nn.init as init
import numpy as np
from models.spec_conv import *

BatchNorm2d = nn.BatchNorm2d
BN_MOMENTUM = 0.01

class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size=3, padding=1, stride=1):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, stride=stride)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out

class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        use_gt            : whether use the ground truth label map to compute the similarity map
        fetch_attention   : whether return the estimated similarity map
    Return:
        N X C X H X W
    '''
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 use_gt=False,
                 use_bg=False,
                 fetch_attention=False,
                 save_grad=False):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.use_gt = use_gt
        self.use_bg = use_bg
        self.save_grad = save_grad
        self.fetch_attention = fetch_attention
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            BatchNorm2d(self.key_channels, momentum=BN_MOMENTUM),
            nn.ReLU(),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            BatchNorm2d(self.key_channels, momentum=BN_MOMENTUM),
            nn.ReLU(),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0),
            BatchNorm2d(self.key_channels, momentum=BN_MOMENTUM),
            nn.ReLU(),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                kernel_size=1, stride=1, padding=0),
            BatchNorm2d(self.in_channels, momentum=BN_MOMENTUM),
            nn.ReLU(),
        )
        self.gradients = []
        #self.shape = ()

    def get_gradients(self):
        return self.gradients

    def save_gradients(self,grad):
        #b,_,c = grad.size()
        #print(torch.sum(grad))
        self.gradients = [] # Clear previous batch gradients
        self.gradients.append(grad)

    def forward(self, x, proxy, gt_label=None):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        # Query of pixels
        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1) # B x N x C
        # Key of object regions
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1) # B x C x K
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1) # B x N x C

        if self.use_gt and gt_label is not None:
            gt_label = label_to_onehot(gt_label.squeeze(1).type(torch.cuda.LongTensor), proxy.size(2)-1)
            sim_map = gt_label[:, :, :, :].permute(0, 2, 3, 1).view(batch_size, h*w, -1) # Probability as sim map
            if self.use_bg:
                bg_sim_map = 1.0 - sim_map # What's not object
                bg_sim_map = F.normalize(bg_sim_map, p=1, dim=-1)
            sim_map = F.normalize(sim_map, p=1, dim=-1)
        else:
            sim_map = torch.matmul(query, key)
            sim_map = (self.key_channels**-.5) * sim_map
            sim_map = F.softmax(sim_map, dim=-1)

        if self.save_grad:
            #self.shape = (h,w)
            sim_map_reshape = sim_map.reshape(batch_size,h,w,-1)
            sim_map_reshape.register_hook(self.save_gradients)
            sim_map = sim_map_reshape.reshape(batch_size,h*w,-1)
            #sim_map.register_hook(self.save_gradients)
        # add bg context ...
        context = torch.matmul(sim_map, value) # hw x k x k x c
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=True)

        if self.use_bg:
            bg_context = torch.matmul(bg_sim_map, value)
            bg_context = bg_context.permute(0, 2, 1).contiguous()
            bg_context = bg_context.view(batch_size, self.key_channels, *x.size()[2:])
            bg_context = self.f_up(bg_context)
            bg_context = F.interpolate(input=bg_context, size=(h, w), mode='bilinear', align_corners=True)
            return context, bg_context
        else:
            if self.fetch_attention:
                sim_map = sim_map.reshape(batch_size, h * w, -1).transpose(1, 2)
                sim_map = sim_map.reshape(batch_size, -1, h, w)
                return context, sim_map
            else:
                return context

class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 use_gt=False,
                 use_bg=False,
                 fetch_attention=False,
                 save_grad=False):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,
                                                     key_channels,
                                                     scale,
                                                     use_gt,
                                                     use_bg,
                                                     fetch_attention,save_grad)


def label_to_onehot(gt, num_classes, ignore_index=-1):
    '''
    gt: ground truth with size (N, H, W)
    num_classes: the number of classes of different label
    '''
    N, H, W = gt.size()
    x = gt
    x[x == ignore_index] = num_classes
    # convert label into onehot format
    onehot = torch.zeros(N, x.size(1), x.size(2), num_classes + 1).cuda()
    onehot = onehot.scatter_(-1, x.unsqueeze(-1), 1)

    return onehot.permute(0, 3, 1, 2)

class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=1, use_gt=False):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale
        self.use_gt = use_gt

    def forward(self, feats, probs, gt_probs=None):
        if self.use_gt and gt_probs is not None:
            gt_probs = label_to_onehot(gt_probs.squeeze(1).type(torch.cuda.LongTensor), probs.size(1))
            batch_size, c, h, w = gt_probs.size(0), gt_probs.size(1), gt_probs.size(2), gt_probs.size(3)
            gt_probs = gt_probs.view(batch_size, c, -1) # B x C x (H*W)
            feats = feats.view(batch_size, feats.size(1), -1) # B x C x H
            feats = feats.permute(0, 2, 1) # B x H x C
            gt_probs = F.normalize(gt_probs, p=1, dim=2) # batch x k x hw
            ocr_context = torch.matmul(gt_probs, feats).permute(0, 2, 1).unsqueeze(3) # batch x k x c
            return ocr_context
        else:
            if feats.size() != probs.size():
                probs = F.upsample(probs,size=(feats.size(-2),feats.size(-1)))
            batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
            probs = probs.view(batch_size, c, -1)
            feats = feats.view(batch_size, feats.size(1), -1)
            feats = feats.permute(0, 2, 1) # batch x hw x c
            probs = F.softmax(self.scale * probs, dim=2)# batch x k x hw
            ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)# batch x k x c
            return ocr_context


class SpatialOCR_Context(nn.Module):
    """
    Implementation of the FastOC module:
    We aggregate the global object representation to update the representation for each pixel.
    """

    def __init__(self, in_channels, key_channels, scale=1, dropout=0.0, bn_type=None, ):
        super(SpatialOCR_Context, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels,
                                                           key_channels,
                                                           scale)

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)
        return context

class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.

    use_gt=True: whether use the ground-truth label to compute the ideal object contextual representations.
    use_bg=True: use the ground-truth label to compute the ideal background context to augment the representations.
    use_oc=True: use object context or not.
    """
    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 scale=1,
                 dropout=0.1,
                 use_gt=False,
                 use_bg=False,
                 use_oc=True,
                 fetch_attention=False,
                 save_grad=False):
        super(SpatialOCR_Module, self).__init__()
        self.use_gt = use_gt
        self.use_bg = use_bg
        self.use_oc = use_oc
        self.fetch_attention = fetch_attention
        self.object_context_block = ObjectAttentionBlock2D(in_channels,
                                                           key_channels,
                                                           scale,
                                                           use_gt,
                                                           use_bg,
                                                           fetch_attention,
                                                           save_grad=save_grad)
        if self.use_bg:
            if self.use_oc:
                _in_channels = 3 * in_channels
            else:
                _in_channels = 2 * in_channels
        else:
            _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0),
            BatchNorm2d(out_channels, momentum=BN_MOMENTUM),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats, gt_label=None):
        if self.use_gt and gt_label is not None:
            if self.use_bg:
                context, bg_context = self.object_context_block(feats, proxy_feats, gt_label)
            else:
                context = self.object_context_block(feats, proxy_feats, gt_label)
        else:
            if self.fetch_attention:
                context, sim_map = self.object_context_block(feats, proxy_feats)
            else:
                context = self.object_context_block(feats, proxy_feats)

        if self.use_bg:
            if self.use_oc:
                output = self.conv_bn_dropout(torch.cat([context, bg_context, feats], 1))
            else:
                output = self.conv_bn_dropout(torch.cat([bg_context, feats], 1))
        else:
            output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        if self.fetch_attention:
            return output, sim_map
        else:
            return output

class AttentionOCR(nn.Module):
    def __init__(self, num_classes, in_channels, hidden_channels, out_channels,
                 scale=1, use_gt=False,
                 dropout=0.05, fetch_attention=True,
                 save_grad=False):
        super(AttentionOCR, self).__init__()
        self.use_gt = use_gt
        self.spatial_context_head = SpatialGather_Module(num_classes, use_gt=use_gt)
        self.fetch_attention = fetch_attention
        self.spatial_ocr_head = SpatialOCR_Module(in_channels=in_channels,
                                                  key_channels=hidden_channels,
                                                  out_channels=out_channels,
                                                  scale=scale,
                                                  dropout=dropout,
                                                  fetch_attention=fetch_attention,
                                                  save_grad=save_grad
                                                  )
    def get_attention_grad(self):
        return self.spatial_ocr_head.object_context_block.get_gradients()

    def forward(self,x,class_centroids,label=None):
        if self.use_gt and label is not None:
            label = F.interpolate(input=label.unsqueeze(1).type(torch.cuda.FloatTensor), size=(x.size(2), x.size(3)),
                                  mode="nearest")

        context = self.spatial_context_head(x, class_centroids, gt_probs=label)
        x = self.spatial_ocr_head(x, context)
        return x


class AttentionConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(AttentionConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.rel_h = nn.Parameter(torch.randn(out_channels // 2, 1, 1, kernel_size, 1), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn(out_channels // 2, 1, 1, 1, kernel_size), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])
        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = self.value_conv(padded_x)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)

        k_out_h, k_out_w = k_out.split(self.out_channels // 2, dim=1)
        k_out = torch.cat((k_out_h + self.rel_h, k_out_w + self.rel_w), dim=1)

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        out = F.softmax(out, dim=-1)
        out = torch.einsum('bnchwk,bnchwk -> bnchw', out, v_out).view(batch, -1, height, width)

        return out

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.rel_h, 0, 1)
        init.normal_(self.rel_w, 0, 1)

class AugmentedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dk, dv, Nh, shape=0, relative=False, stride=1):
        super(AugmentedConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.dk = dk
        self.dv = dv
        self.Nh = Nh
        self.shape = shape
        self.relative = relative
        self.stride = stride
        self.padding = (self.kernel_size - 1) // 2

        assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
        assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
        assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
        assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."

        self.conv_out = nn.Conv2d(self.in_channels, self.out_channels - self.dv, self.kernel_size, stride=stride, padding=self.padding)

        self.qkv_conv = nn.Conv2d(self.in_channels, 2 * self.dk + self.dv, kernel_size=self.kernel_size, stride=stride, padding=self.padding)

        self.attn_out = nn.Conv2d(self.dv, self.dv, kernel_size=1, stride=1)

        if self.relative:
            self.key_rel_w = nn.Parameter(torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True))
            self.key_rel_h = nn.Parameter(torch.randn((2 * self.shape - 1, dk // Nh), requires_grad=True))

    def forward(self, x):
        # Input x
        # (batch_size, channels, height, width)
        # batch, _, height, width = x.size()

        # conv_out
        # (batch_size, out_channels, height, width)
        conv_out = self.conv_out(x)
        batch, _, height, width = conv_out.size()

        # flat_q, flat_k, flat_v
        # (batch_size, Nh, height * width, dvh or dkh)
        # dvh = dv / Nh, dkh = dk / Nh
        # q, k, v
        # (batch_size, Nh, height, width, dv or dk)
        flat_q, flat_k, flat_v, q, k, v = self.compute_flat_qkv(x, self.dk, self.dv, self.Nh)
        logits = torch.matmul(flat_q.transpose(2, 3), flat_k)
        if self.relative:
            h_rel_logits, w_rel_logits = self.relative_logits(q)
            logits += h_rel_logits
            logits += w_rel_logits
        weights = F.softmax(logits, dim=-1)

        # attn_out
        # (batch, Nh, height * width, dvh)
        attn_out = torch.matmul(weights, flat_v.transpose(2, 3))
        attn_out = torch.reshape(attn_out, (batch, self.Nh, self.dv // self.Nh, height, width))
        # combine_heads_2d
        # (batch, out_channels, height, width)
        attn_out = self.combine_heads_2d(attn_out)
        attn_out = self.attn_out(attn_out)
        return torch.cat((conv_out, attn_out), dim=1), weights

    def compute_flat_qkv(self, x, dk, dv, Nh):
        qkv = self.qkv_conv(x)
        N, _, H, W = qkv.size()
        q, k, v = torch.split(qkv, [dk, dk, dv], dim=1)
        q = self.split_heads_2d(q, Nh)
        k = self.split_heads_2d(k, Nh)
        v = self.split_heads_2d(v, Nh)

        dkh = dk // Nh
        q *= dkh ** -0.5
        flat_q = torch.reshape(q, (N, Nh, dk // Nh, H * W))
        flat_k = torch.reshape(k, (N, Nh, dk // Nh, H * W))
        flat_v = torch.reshape(v, (N, Nh, dv // Nh, H * W))
        return flat_q, flat_k, flat_v, q, k, v

    def split_heads_2d(self, x, Nh):
        batch, channels, height, width = x.size()
        ret_shape = (batch, Nh, channels // Nh, height, width)
        split = torch.reshape(x, ret_shape)
        return split

    def combine_heads_2d(self, x):
        batch, Nh, dv, H, W = x.size()
        ret_shape = (batch, Nh * dv, H, W)
        return torch.reshape(x, ret_shape)

    def relative_logits(self, q):
        B, Nh, dk, H, W = q.size()
        q = torch.transpose(q, 2, 4).transpose(2, 3)

        rel_logits_w = self.relative_logits_1d(q, self.key_rel_w, H, W, Nh, "w") # B x Nh x H*W x H*W
        rel_logits_h = self.relative_logits_1d(torch.transpose(q, 2, 3), self.key_rel_h, W, H, Nh, "h")

        return rel_logits_h, rel_logits_w

    def relative_logits_1d(self, q, rel_k, H, W, Nh, case):
        rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k) # B x Nh x H x W x (2W-1)
        rel_logits = torch.reshape(rel_logits, (-1, Nh * H, W, 2 * W - 1))
        rel_logits = self.rel_to_abs(rel_logits)

        rel_logits = torch.reshape(rel_logits, (-1, Nh, H, W, W)) # B x Nh x H x W x W
        rel_logits = torch.unsqueeze(rel_logits, dim=3)
        rel_logits = rel_logits.repeat((1, 1, 1, H, 1, 1))

        if case == "w":
            rel_logits = torch.transpose(rel_logits, 3, 4)
        elif case == "h":
            rel_logits = torch.transpose(rel_logits, 2, 4).transpose(4, 5).transpose(3, 5)
        rel_logits = torch.reshape(rel_logits, (-1, Nh, H * W, H * W))
        return rel_logits

    def rel_to_abs(self, x):
        B, Nh, L, _ = x.size()

        col_pad = torch.zeros((B, Nh, L, 1)).to(x)
        x = torch.cat((x, col_pad), dim=3)

        flat_x = torch.reshape(x, (B, Nh, L * 2 * L))
        flat_pad = torch.zeros((B, Nh, L - 1)).to(x)
        flat_x_padded = torch.cat((flat_x, flat_pad), dim=2) # B x Nh*H x 2W^2 + W - 1

        final_x = torch.reshape(flat_x_padded, (B, Nh, L + 1, 2 * L - 1))
        final_x = final_x[:, :, :L, L - 1:]
        return final_x

class AttentionStem(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, m=4, bias=False):
        super(AttentionStem, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.m = m

        assert self.out_channels % self.groups == 0, "out_channels should be divided by groups. (example: out_channels: 40, groups: 4)"

        self.emb_a = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_b = nn.Parameter(torch.randn(out_channels // groups, kernel_size), requires_grad=True)
        self.emb_mix = nn.Parameter(torch.randn(m, out_channels // groups), requires_grad=True)

        self.key_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.query_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.value_conv = nn.ModuleList([nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias) for _ in range(m)])

        self.reset_parameters()

    def forward(self, x):
        batch, channels, height, width = x.size()

        padded_x = F.pad(x, [self.padding, self.padding, self.padding, self.padding])

        q_out = self.query_conv(x)
        k_out = self.key_conv(padded_x)
        v_out = torch.stack([self.value_conv[_](padded_x) for _ in range(self.m)], dim=0)

        k_out = k_out.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        v_out = v_out.unfold(3, self.kernel_size, self.stride).unfold(4, self.kernel_size, self.stride)

        k_out = k_out[:, :, :height, :width, :, :]
        v_out = v_out[:, :, :, :height, :width, :, :]

        emb_logit_a = torch.einsum('mc,ca->ma', self.emb_mix, self.emb_a)
        emb_logit_b = torch.einsum('mc,cb->mb', self.emb_mix, self.emb_b)
        emb = emb_logit_a.unsqueeze(2) + emb_logit_b.unsqueeze(1)
        emb = F.softmax(emb.view(self.m, -1), dim=0).view(self.m, 1, 1, 1, 1, self.kernel_size, self.kernel_size)

        v_out = emb * v_out

        k_out = k_out.contiguous().view(batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = v_out.contiguous().view(self.m, batch, self.groups, self.out_channels // self.groups, height, width, -1)
        v_out = torch.sum(v_out, dim=0).view(batch, self.groups, self.out_channels // self.groups, height, width, -1)

        q_out = q_out.view(batch, self.groups, self.out_channels // self.groups, height, width, 1)

        out = q_out * k_out
        att = F.softmax(out, dim=-1)
        #print(att.shape)
        out = torch.einsum('bnchwk,bnchwk->bnchw', att, v_out).view(batch, -1, height, width)
        att = att.transpose(1,5).squeeze(-1)
        return out, att.contiguous().view(batch,-1,height,width)

    def reset_parameters(self):
        init.kaiming_normal_(self.key_conv.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_normal_(self.query_conv.weight, mode='fan_out', nonlinearity='relu')
        for _ in self.value_conv:
            init.kaiming_normal_(_.weight, mode='fan_out', nonlinearity='relu')

        init.normal_(self.emb_a, 0, 1)
        init.normal_(self.emb_b, 0, 1)
        init.normal_(self.emb_mix, 0, 1)



class BottleNeck(nn.Module):
    def __init__(self,c_in, c_out, stride=1, kernel_size=3, negative_slope=0.2,
               do_upsample=False, do_batchnorm=False,rates=(2,4,8)):
        super(BottleNeck, self).__init__()
        # compute the padding amount
        padding = int(np.ceil(((kernel_size - 1) + 1 - stride) / 2.))
        self.sa = SAModule(channels=c_in,rates=rates)
        self.conv_connect = nn.Conv2d(c_in,c_out,kernel_size=1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(negative_slope)
        )

        # append an Upsample layer if it is required
        if do_upsample:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        else:
            self.up = None

        # append an BatchNormalization layer if it is required
        if do_batchnorm:
            self.bn = nn.BatchNorm2d(c_out)
        else:
            self.bn = None

    def forward(self,x):
        x = self.sa(x)
        x1 = self.conv1(x)
        if self.up is not None:
            x1 = self.up(x1)
        x = self.conv_connect(x)
        x = F.upsample(x,(x1.size(-2),x1.size(-1)))
        x1 = x1 + x

        if self.bn is not None:
            x1 = self.bn(x1)
        return x1

class DilationBottleNeck(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, kernel_size=3, do_upsample=False):
        super(DilationBottleNeck, self).__init__()
        # compute the padding amount
        dilation=1
        padding = int(np.ceil((dilation*(kernel_size - 1) + 1 - stride) / 2.))
        self.block1 = nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.Conv2d(in_planes, planes, kernel_size=kernel_size, padding=padding, bias=True,stride=stride,dilation=dilation),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate)
        )
        self.shortcut12 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, bias=True,padding=padding)
        self.shortcut13 = nn.Conv2d(in_planes, planes, kernel_size=kernel_size, stride=stride, bias=True,
                                    padding=padding)

        dilation = 2
        padding = int(np.ceil((dilation*(kernel_size - 1) + 1 - stride) / 2.))
        self.block2 = nn.Sequential(
            nn.BatchNorm2d(planes),
            nn.Conv2d(planes, planes, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=True),
            nn.ReLU()
        )
        self.do_upsample = do_upsample
        # self.conv1 =
        # self.dropout = nn.Dropout(p=dropout_rate)
        # self.bn2 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)



    def forward(self, x):
        out2 = self.block1(x)
        out2 += self.shortcut12(x)
        out3 = self.block2(out2)
        out3 += out2
        out3 += self.shortcut13(x)
        if self.do_upsample:
            out3 = F.upsample(out3,scale_factor=2, mode='bilinear', align_corners=False)
        return out3

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer = None,
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

def conv_block(c_in, c_out, stride=1, kernel_size=3, negative_slope=0.2,
               do_upsample=False, do_batchnorm=False):
    """
    Creates a convolutional building block: Conv + LeakyReLU + Upsample (optional) + BatchNorm (optional)

    :param c_in: input channel size
    :param c_out: output channel size
    :param kernel_size: filter size of the conv layer
    :param stride: stride of the convolutional layer
    :param negative_slope: the parameter that controls the angle of the negative slope of the LeakyReLU layer
    :param do_upsample: a boolean param indicating whether an upsample layer is added after the (Conv + LeakyReLU)
    :param do_batchnorm: a boolean param indicating whether an upsample layer is added at the end of the block
    :return: a convolutional building block
    """

    block = nn.ModuleList()
    # compute the padding amount
    padding = int(np.ceil(((kernel_size - 1) + 1 - stride) / 2.))
    #block.append(depthwise_separable_conv(c_in,c_out,kernel_size=kernel_size,padding=padding,stride=stride))
    block.append(nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding))
    block.append(nn.LeakyReLU(negative_slope))

    # append an Upsample layer if it is required
    if do_upsample:
        block.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))

    # append an BatchNormalization layer if it is required
    if do_batchnorm:
        block.append(nn.BatchNorm2d(c_out))

    return nn.Sequential(*block)