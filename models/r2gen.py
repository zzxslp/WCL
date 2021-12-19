import torch
import torch.nn as nn
import numpy as np
import sys

from modules.visual_extractor import VisualExtractor
from modules.encoder_decoder import EncoderDecoder
from modules.projection import Projection


class R2GenModel(nn.Module):
    def __init__(self, args, tokenizer):
        super(R2GenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        # if args.dataset_name == "iu_xray" or args.dataset_name == "mimic_cxr":
        self.visual_extractor = VisualExtractor(args)
        self.encoder_decoder = EncoderDecoder(args, tokenizer)
        self.projection = Projection(args.contra_embed_size)
        if args.dataset_name == 'iu_xray':
            self.forward = self.forward_iu_xray
        elif args.dataset_name == 'mimic_cxr':
            self.forward = self.forward_mimic_cxr
        elif args.dataset_name == 'mimic_abn':
            self.forward = self.forward_mimic_abn

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_mimic_abn(self, images, targets=None, mode='train'):
        img1, img2 = images[:, 0], images[:, 1]
        img1, img2 = img1.reshape(img1.size(0), img1.size(1), -1).permute(0, 2, 1), img2.reshape(img2.size(0), img2.size(1), -1).permute(0, 2, 1) # # (16, 16, 1024)
        att_feats = torch.cat((img1, img2), dim=1) 
        fc_feats = torch.cat((img1.mean(dim=1).squeeze(), img2.mean(dim=1).squeeze()), dim=1) 
        if mode == 'train':           
            outputs = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return outputs
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
            return output

    def forward_iu_xray(self, images, targets=None, mode='train'):
        att_feats_0, fc_feats_0 = self.visual_extractor(images[:, 0]) # features for attn and fc 
        att_feats_1, fc_feats_1 = self.visual_extractor(images[:, 1]) 
        fc_feats = torch.cat((fc_feats_0, fc_feats_1), dim=1)
        att_feats = torch.cat((att_feats_0, att_feats_1), dim=1)
        if mode == 'train':
            outputs= self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
            return outputs
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
            return output
        # else:
        #     raise ValueError

    def forward_mimic_cxr(self, images, targets=None, mode='train'):
        att_feats, fc_feats = self.visual_extractor(images)
        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        else:
            raise ValueError
        return output

