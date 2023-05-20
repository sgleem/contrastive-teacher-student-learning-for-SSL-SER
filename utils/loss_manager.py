import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import torch.autograd as autograd
from collections import defaultdict

class LogManager:
    def __init__(self):
        self.log_book=defaultdict(lambda: [])
    def alloc_stat_type(self, stat_type):
        self.log_book[stat_type] = []
    def alloc_stat_type_list(self, stat_type_list):
        for stat_type in stat_type_list:
            self.alloc_stat_type(stat_type)
    def init_stat(self):
        for stat_type in self.log_book.keys():
            self.log_book[stat_type] = []
    def add_stat(self, stat_type, stat):
        assert stat_type in self.log_book, "Wrong stat type"
        self.log_book[stat_type].append(stat)
    def add_torch_stat(self, stat_type, stat):
        assert stat_type in self.log_book, "Wrong stat type"
        self.log_book[stat_type].append(stat.detach().cpu().item())
    def get_stat(self, stat_type):
        result_stat = 0
        stat_list = self.log_book[stat_type]
        if len(stat_list) != 0:
            result_stat = np.mean(stat_list)
            result_stat = np.round(result_stat, 4)
        return result_stat

    def print_stat(self):
        for stat_type in self.log_book.keys():
            if len(self.log_book[stat_type]) == 0:
                continue
            stat = self.get_stat(stat_type)           
            print(stat_type,":",stat, end=' / ')
        print(" ")

    def get_stat_str(self):
        result_str = ""
        for stat_type in self.log_book.keys():
            if len(self.log_book[stat_type]) == 0:
                continue
            stat = self.get_stat(stat_type)           
            result_str += str(stat) + " / "
        return result_str

def CCC_loss(pred, lab, m_lab=None, v_lab=None, is_numpy=False):
    """
    pred: (N, 3)
    lab: (N, 3)
    """
    if is_numpy:
        pred = torch.Tensor(pred).float().cuda()
        lab = torch.Tensor(lab).float().cuda()
    
    m_pred = torch.mean(pred, 0, keepdim=True)
    m_lab = torch.mean(lab, 0, keepdim=True)

    d_pred = pred - m_pred
    d_lab = lab - m_lab

    v_pred = torch.var(pred, 0, unbiased=False)
    v_lab = torch.var(lab, 0, unbiased=False)

    corr = torch.sum(d_pred * d_lab, 0) / (torch.sqrt(torch.sum(d_pred ** 2, 0)) * torch.sqrt(torch.sum(d_lab ** 2, 0)))

    s_pred = torch.std(pred, 0, unbiased=False)
    s_lab = torch.std(lab, 0, unbiased=False)

    ccc = (2*corr*s_pred*s_lab) / (v_pred + v_lab + (m_pred[0]-m_lab[0])**2)    
    return ccc

def ladder_loss(decoder_h, encoder_h, layer_wise=False):
    assert len(decoder_h) == len(encoder_h)

    h_num = len(decoder_h)
    if layer_wise:
        total_loss=[torch.zeros(1).float().cuda() for h in range(h_num)]
    else:
        total_loss = torch.zeros(1).float().cuda()
    for h_idx in range(h_num):
        if layer_wise:
            total_loss[h_idx] += F.mse_loss(decoder_h[h_idx], encoder_h[h_num-1-h_idx])
        else:
            total_loss += F.mse_loss(decoder_h[h_idx], encoder_h[h_num-1-h_idx])
    return total_loss

def decoupled_ladder_loss(decoder_he, encoder_he, decoder_hr, encoder_hr):
    assert len(decoder_he) == len(encoder_he) == len(decoder_hr) == len(encoder_hr)

    h_num = len(decoder_he)
    total_loss = torch.zeros(1).float().cuda()
    for h_idx in range(h_num):
        if h_idx == h_num - 1:
            x = encoder_he[h_num-h_idx-1]
            recon_x = decoder_he[h_idx]+decoder_hr[h_idx]
            total_loss += F.mse_loss(recon_x, x)
        else:
            total_loss += F.mse_loss(decoder_he[h_idx], encoder_hr[h_num-h_idx-1])
            total_loss += F.mse_loss(decoder_he[h_idx], encoder_hr[h_num-h_idx-1])
    return total_loss


def orthogonal_loss(eh, rh, eps=1e-10):
    batch_size = eh.size(0)
    out = torch.zeros(1).cuda()
    
    for e, r in zip(eh, rh):
        len_e = torch.sqrt(torch.sum(torch.pow(e, 2)))
        len_r = torch.sqrt(torch.sum(torch.pow(r, 2)))
        out += torch.dot(e, r) / ((len_e*len_r)+eps)
    out /= batch_size
    out = torch.abs(out)
    return out

def orthogonal_loss_2D(eh, rh, eps=1e-10):
    batch_size = eh.size(0)
    out = torch.zeros(1).cuda()
    total_dot = eh*rh
    total_dot = torch.sum(total_dot, dim=2)
    
    len_e = torch.sqrt(torch.sum(torch.pow(eh, 2), dim=2))
    len_r = torch.sqrt(torch.sum(torch.pow(rh, 2), dim=2))

    denom = (len_e*len_r)+eps
    
    total_cos = total_dot / denom
    
    avg_total_cos = torch.mean(torch.abs(total_cos), dim=1)
    # avg_total_cos = torch.mean(total_cos, dim=1)
    out = torch.mean(avg_total_cos)
    
    assert (out >= 0.0) and (out <= 1.0)
    return out

def MSE_emotion(pred, lab):
    aro_loss = F.mse_loss(pred[:][0], lab[:][0])
    dom_loss = F.mse_loss(pred[:][1], lab[:][1])
    val_loss = F.mse_loss(pred[:][2], lab[:][2])

    return [aro_loss, dom_loss, val_loss]

def CE_category(pred, lab):
    return F.cross_entropy(pred, lab)

def NLL_category(pred, lab):
    return nn.NLLLoss()(pred, lab)

def calc_err(pred, lab):
    p = pred.detach()
    t = lab.detach()
    total_num = p.size()[0]
    ans = torch.argmax(p, dim=1)
    corr = torch.sum((ans==t).long())

    err = (total_num-corr) / total_num

    return err

def calc_acc(pred, lab):
    err = calc_err(pred, lab)
    return 1.0 - err

def self_entropy(log_prob):
    prob = torch.exp(log_prob)
    b = prob * torch.log2(prob)
    b = torch.mean(-1.0 * b.sum(dim=1))
    return b

def calc_rank_loss(pair_set, rank_scores):
    batch_len = len(pair_set)
    loss = torch.zeros(1).cuda()
    for higher_idx, lower_idx in pair_set:
        score_higher = rank_scores[higher_idx]
        score_lower = rank_scores[lower_idx]

        loss += torch.log(1+torch.exp(-1*(score_higher-score_lower)))
    loss /= batch_len
    return loss

def calc_gradient_penalty(netD, real_data, fake_data):
    #print real_data.size()
    alpha = torch.rand_like(real_data)
    # alpha = torch.ones_like(real_data)
    # alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients = gradients + 1e-16
    gradient_penalty = ((gradients.norm(2, dim=(1,2)) - 1) ** 2).mean()
    # gradient_penalty = gradient_penalty.float()
    return gradient_penalty

def calc_moving_average(pre_ma, cur_ma, gamma=0.99):
    cur_val = torch.mean(cur_ma)
    result_ma = gamma * pre_ma + (1-gamma) * cur_val
    return result_ma

def ctc_loss(pred, lab, audio_mask, lab_mask):
    pred_mask = torch.div(torch.sum(audio_mask, dim=1), 320, rounding_mode='floor') - 1
    pred_mask = pred_mask.long()
    pred = pred.permute(1,0,2)
    with torch.backends.cudnn.flags(enabled=False):
        loss = F.ctc_loss(pred, lab, pred_mask, lab_mask)
    return loss

def gaussian_kernel(a, b):
    dim1_1, dim1_2 = a.shape[0], b.shape[0]
    depth = a.shape[1]
    a = a.view(dim1_1, 1, depth)
    b = b.view(1, dim1_2, depth)
    a_core = a.expand(dim1_1, dim1_2, depth)
    b_core = b.expand(dim1_1, dim1_2, depth)
    numerator = (a_core - b_core).pow(2).mean(2)/depth
    # numerator = (a_core - b_core).pow(2).sum(2) #mean(2)/depth
    # print(numerator.shape)
    return torch.exp(-numerator)


def MMD(a, b):
    return gaussian_kernel(a, a).mean() + gaussian_kernel(b, b).mean() - 2*gaussian_kernel(a, b).mean()