import torch
import torch.nn as nn

class Projection(nn.Module):
    def __init__(self, contra_embed_size):
        super(Projection, self).__init__()
        embed_size = contra_embed_size
        self.src_projection = nn.Sequential(nn.Linear(512, embed_size),
                                   nn.ReLU(),
                                   nn.Linear(embed_size, embed_size))
        self.tgt_projection = nn.Sequential(nn.Linear(512, embed_size),
                                    nn.ReLU(),
                                    nn.Linear(embed_size, embed_size))
        
    def forward(self, src_embed, tgt_embed, tgt_embed_adv=None):
        src_embed = self._normalize(self.src_projection(src_embed.mean(dim=1)))
        tgt_embed = self._normalize(self.tgt_projection(tgt_embed.mean(dim=1)))
        if tgt_embed_adv is not None:
            tgt_embed_adv = self._normalize(self.tgt_projection(tgt_embed_adv.mean(dim=1)))
        else:
            tgt_embed_adv = None
        return src_embed, tgt_embed, tgt_embed_adv

    def _normalize(self, feature):
        norm = feature.norm(p=2, dim=1, keepdim=True)
        feature = feature.div(norm + 1e-16)
        return feature
