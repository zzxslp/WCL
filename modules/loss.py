import torch
import torch.nn as nn
from torch.autograd import Variable, grad
import numpy as np

import sys

class LanguageModelCriterion(nn.Module):
    def __init__(self):
        super(LanguageModelCriterion, self).__init__()

    def forward(self, input, target, mask):
        # truncate to the same size
        target = target[:, :input.size(1)]
        mask = mask[:, :input.size(1)]
        output = -input.gather(2, target.long().unsqueeze(2)).squeeze(2) * mask
        output = torch.sum(output) / torch.sum(mask)

        return output

class ContraLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf."""
    def __init__(self, temperature=1, base_temperature=1):
        super(ContraLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, src_feature, tgt_feature, labels=None, adv_feature=None):
        """
        both input should be l2 normalized
        Args:
            src_feature: (mb, dim)
            tgt_feature: (mb, dim)
            mask: contrastive mask of shape (mb, mb)
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if src_feature.is_cuda
                  else torch.device('cpu'))

        batch_size = src_feature.shape[0]

        labels = labels.unsqueeze(0) # [mb] to [1, mb]

        # add weights for the same cluster
        pos_mask = torch.eye(batch_size).float().to(device)
        mask = 1 * (pos_mask-torch.eq(labels, labels.T).float())

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(src_feature, tgt_feature.T),
            self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        logits_mask = torch.ones_like(mask) - mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask #

        if adv_feature is not None:
            adv_logits = torch.exp((src_feature*adv_feature).sum(1, keepdim=True) - logits_max.detach()) 
            exp_logits = exp_logits.sum(1, keepdim=True) + adv_logits
        else:
            exp_logits = exp_logits.sum(1, keepdim=True)
        log_prob = logits - torch.log(exp_logits)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1)
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss


def compute_loss(model, output, reports_ids, reports_masks, reports_labels, epoch, args):
    (output, src_embed, tgt_embed) = output
    ## hyper-parameters to tune
    adv_epsilon = args.adv_epsilon
    contra_type = args.contra_type
    contra_lambda = args.contra_lambda
    temperature = args.temperature
    contra_epoch = args.contra_epoch
    finetune_lambda = args.finetune_lambda

    criterion = LanguageModelCriterion()
    criterion_contra = ContraLoss(temperature)
    loss_ce = criterion(output, reports_ids[:, 1:], reports_masks[:, 1:]).mean()
    
    if epoch > contra_epoch:
        if contra_type == 'base':
            src_embed, tgt_embed, _ = model.projection(src_embed, tgt_embed)
            loss_contra = criterion_contra(src_embed, tgt_embed, reports_labels)
        loss = finetune_lambda * ((1-contra_lambda)*loss_ce + contra_lambda*loss_contra)
    else:
        loss = loss_ce

    return loss

def _l2_normalize(d):
    if isinstance(d, Variable):
        d = d.data.cpu().numpy()
    elif isinstance(d, torch.FloatTensor) or isinstance(d, torch.cuda.FloatTensor):
        d = d.cpu().numpy()
    d /= (np.sqrt(np.sum(d ** 2, axis=(1, 2))).reshape((-1, 1, 1)) + 1e-16)
    return torch.from_numpy(d)

