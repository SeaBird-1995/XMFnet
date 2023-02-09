'''
Copyright (c) 2023 by Haiming Zhang. All Rights Reserved.

Author: Haiming Zhang
Date: 2023-02-07 18:55:02
Email: haimingzhang@link.cuhk.edu.cn
Description: 
'''

import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import numpy as np
from datetime import datetime, timedelta
from decoder.dec_net import Decoder_Network
from encoder_dgcnn.dgcnn import DGCNN
from encoder_image.resnet import ResNet
from model import Network
from config import params

p = params()


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, batch_first=True) -> None:
        super().__init__()
        self.multi_head_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=batch_first)
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, queries, keys, values):
        x, _ = self.multi_head_attn(queries, keys, values)
        feat = self.layer_norm(x + queries) # B x N x F
        return feat
    

class MultiLevelEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, batch_first=True):
        
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(embed_dim, num_heads, batch_first)
                                     for _ in range(num_layers)])
        
    def forward(self, input):
        outs = []
        out = input
        for layer in self.layers:
            out = layer(out, out, out)
            outs.append(out.unsqueeze(1))

        outs = torch.cat(outs, 1)
        return outs
        

class XModalEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, batch_first=True):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(embed_dim, num_heads, batch_first)
                                     for _ in range(num_layers)])

    def forward(self, input, extra_feat):
        outs = []
        out = input

        for i, l in enumerate(self.layers):
            out = l(out, out, out)
            out = out + extra_feat[:, i]
            outs.append(out.unsqueeze(1))

        outs = torch.cat(outs, 1)
        return outs


class Network3DOnly(nn.Module):
    """The model only using 3D.

    Args:
        nn (_type_): _description_
    """

    def __init__(self):

        super(Network3DOnly, self).__init__()

        # Encoders for images and Point clouds
        self.pc_encoder = DGCNN()
        self.im_encoder = ResNet()

        self.encoder = MultiLevelEncoder(3, p.d_attn, p.num_heads)

        # # Attention layers to fuse the information from the two modalities
        # self.self_attn1 = nn.MultiheadAttention(
        #     p.d_attn, p.num_heads, batch_first=True)
        # self.layer_norm1 = nn.LayerNorm(p.d_attn)

        # self.self_attn2 = nn.MultiheadAttention(
        #     p.d_attn, p.num_heads, batch_first=True)
        # self.layer_norm2 = nn.LayerNorm(p.d_attn)

        # self.self_attn3 = nn.MultiheadAttention(
        #     p.d_attn, p.num_heads, batch_first=True)
        # self.layer_norm3 = nn.LayerNorm(p.d_attn)


        # Decoder Network to reconstruct the point cloud
        self.decoder = Decoder_Network()

    def forward(self, x_part, view):

       
        pc_feat = self.pc_encoder(x_part)  #B x F x N 
        im_feat = self.im_encoder(view)  #B x F x N

        im_feat = im_feat.permute(0, 2, 1)
        pc_feat = pc_feat.permute(0, 2, 1)

        pc_feat = self.encoder(pc_feat)

        # x, _ = self.self_attn1(pc_feat, pc_feat, pc_feat)
        # pc_feat = self.layer_norm1(x + pc_feat) # B x N x F
        
        # x, _ = self.self_attn2(pc_feat, pc_feat, pc_feat)
        # pc_feat = self.layer_norm2(x + pc_feat)

        # x, _ = self.self_attn3(pc_feat, pc_feat, pc_feat)
        # pc_feat = self.layer_norm3(x + pc_feat)


        x_part = x_part.permute(0, 2, 1)  # B x 3 x N ----> B x N x 3
        

        final = self.decoder(pc_feat, x_part)
            
        return final


class NetworkDistill(nn.Module):
    """The model by using distillation technique.
    """

    def __init__(self):

        super(NetworkDistill, self).__init__()

        # Encoders for images and Point clouds
        self.pc_encoder = DGCNN()
        self.im_encoder = ResNet()

        self.encoder_student = MultiLevelEncoder(3, p.d_attn, p.num_heads)

        self.encoder_teacher = XModalEncoder(3, p.d_attn, p.num_heads)

        self.decoder = Decoder_Network()

    def forward(self, x_part, view):
        pc_feat = self.pc_encoder(x_part)  #B x F x N
        im_feat = self.im_encoder(view)  #B x F x N

        im_feat = im_feat.permute(0, 2, 1)
        pc_feat = pc_feat.permute(0, 2, 1)

        feat_student = self.encoder_student(pc_feat)

        feat_teacher = self.encoder_teacher(im_feat, pc_feat)

        ## Decoder
        x_part = x_part.permute(0, 2, 1)  # B x 3 x N ----> B x N x 3
        

        final = self.decoder(pc_feat, x_part)
            
        return final, feat_student, feat_teacher
    

if __name__ == '__main__':
    pc_feat = torch.randn(16, 128, 256).cuda()
    im_feat = torch.randn(16, 196, 256).cuda()

    encoder = MultiLevelEncoder(3, 256, 4).cuda()
    output = encoder(pc_feat)
    print(output.shape)
    exit(0)

   
    x_part = torch.randn(16, 3, 2048).cuda()
    view = torch.randn(16, 3, 224, 224).cuda()
    model = Network3DOnly().cuda()
    out = model(x_part, view)
    print(out.shape)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in model_parameters])
    print(f"n parameters:{parameters}")
    
