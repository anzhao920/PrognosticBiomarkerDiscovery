"""
Train and eval functions used in main.py
"""
import math
import os
import sys
from typing import Iterable
import pandas as pd
import torch

import util.misc as utils
from sksurv.metrics import integrated_brier_score, concordance_index_censored, concordance_index_ipcw
import numpy as np
from torch.utils.tensorboard import SummaryWriter
# import matplotlib.pyplot as plt
from sksurv.linear_model.coxph import BreslowEstimator
from torch.autograd import Variable

def mixup_data(sample, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    (patientEmbedding, pos, keyPaddingMask, cluster, Dead, followUpTime, patientIdx) = sample
    batch_size = patientEmbedding.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    patientEmbedding = lam * patientEmbedding + (1 - lam) * patientEmbedding[index, :]
    pos = lam * pos + (1 - lam) * pos[index, :]
    if lam<0.5:
        keyPaddingMask = keyPaddingMask[index, :]
        cluster = cluster[index, :]
    Dead_a, Dead_b = Dead, Dead[index]
    followUpTime_a, followUpTime_b = followUpTime,followUpTime[index]
    return patientEmbedding, pos, keyPaddingMask, cluster, Dead_a, Dead_b, followUpTime_a,followUpTime_b, lam


def mixup_criterion(criterion, outputs, followUpTime_a,followUpTime_b, Dead_a,Dead_b,lam):

    loss_dict_a = criterion(outputs,followUpTime_a,Dead_a)
    loss_dict_b = criterion(outputs,followUpTime_b,Dead_b)
    loss_dict = {k:lam*loss_dict_a[k] + (1 - lam)*loss_dict_b[k] for k in loss_dict_a}
    return loss_dict


def train_one_epoch_SAM(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, fold: int, tb_writer: SummaryWriter, max_norm: float = 0, logSaveInterval = 500,mixUp=True,SAM=True):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))

    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for batch_idx, sample in metric_logger.log_every(data_loader, print_freq, header):
        sample = [element.to(device) for element in sample]
        if mixUp:
            alpha = 2
            patientEmbedding, pos, keyPaddingMask, cluster, Dead_a, Dead_b, followUpTime_a,followUpTime_b, lam\
            = mixup_data(sample,alpha)
            if sum(Dead_a)==0:
                metric_logger.update(loss=0)
                metric_logger.update(neg_likelihood=0)
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])
                continue
            patientEmbedding, pos, keyPaddingMask, cluster, Dead_a, Dead_b, followUpTime_a,followUpTime_b = \
            map(Variable, (patientEmbedding, pos, keyPaddingMask, cluster, Dead_a, Dead_b, followUpTime_a,followUpTime_b))
            outputs = model(patientEmbedding, pos, keyPaddingMask, cluster)
            loss_dict = mixup_criterion(criterion, outputs, followUpTime_a,followUpTime_b, Dead_a,Dead_b,lam)
        else:
            (patientEmbedding, pos, keyPaddingMask, cluster, Dead, followUpTime, patientIdx) = sample
            if sum(Dead)==0:
                metric_logger.update(loss=0)
                metric_logger.update(neg_likelihood=0)
                metric_logger.update(lr=optimizer.param_groups[0]["lr"])
                continue   
            outputs = model(patientEmbedding, pos, keyPaddingMask, cluster)
            loss_dict = criterion(outputs,followUpTime,Dead)

        weight_dict = criterion.weight_dict
        losses_1 = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())
        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        if SAM:
            optimizer.zero_grad()
            losses_1.backward()        
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.first_step(zero_grad=True)
            
            outputs_2 = model(patientEmbedding, pos, keyPaddingMask, cluster)
            if mixUp:
                loss_dict_2 = mixup_criterion(criterion, outputs_2, followUpTime_a,followUpTime_b, Dead_a,Dead_b,lam)
            else:
                loss_dict_2 = criterion(outputs_2,followUpTime,Dead)
            losses_2 = sum(loss_dict_2[k] * weight_dict[k] for k in loss_dict_2.keys() if k in weight_dict)
            losses_2.backward()
            # second forward-backward pass
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)        
            optimizer.second_step(zero_grad=True)
        else:
            optimizer.zero_grad()
            losses_1.backward()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()            

        metric_logger.update(loss=loss_value)
        metric_logger.update(neg_likelihood=loss_dict_reduced['neg_likelihood'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, data_loader, device, output_dir,status):
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = status+' evaluate:'
    event_indicator = np.asarray([],dtype=bool)
    event_time = np.asarray([])
    estimate = np.asarray([])
    A = []
    print_freq = 10

    for batch_id, sample in metric_logger.log_every(data_loader, print_freq, header):
        sample = [element.to(device) for element in sample]
        (patientEmbedding, pos, keyPaddingMask, cluster, Dead, followUpTime,patientIdx) = sample
        if sum(Dead)==0:
            continue
        event_indicator=np.append(event_indicator,Dead.cpu().numpy().astype(bool))
        event_time=np.append(event_time,followUpTime.cpu().numpy())  
        outputs = model(patientEmbedding, pos, keyPaddingMask, cluster)
        estimate = np.append(estimate,outputs[0].cpu().numpy())
        A.append(outputs[2].cpu().numpy())

        loss_dict = criterion(outputs,followUpTime,Dead)
        weight_dict = criterion.weight_dict
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()))
        metric_logger.update(neg_likelihood=loss_dict_reduced['neg_likelihood'])
        
    CIndex = concordance_index_censored(event_indicator, event_time, estimate)
    metric_logger.synchronize_between_processes()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    stats['CIndex']=CIndex[0]
    return stats

@torch.no_grad()
def test(model, criterion, test_data_loader, train_data_loader, device, output_dir,fold,coxBiomarkerRisk=None):
    model.eval()
    criterion.eval()

    event_indicator_train = np.asarray([],dtype=bool)
    event_time_train = np.asarray([])
    estimate_train = np.asarray([])
    event_indicator_test = np.asarray([],dtype=bool)
    event_time_test = np.asarray([])
    estimate_test = np.asarray([])
    survival_score_patches = np.asarray([])
    all_clusters = np.asarray([])
    print_freq = 10
    max_num_cluster = model.max_num_cluster
    meanClusterRisk = []
    stdClusterRisk = []
    LearnedClusterRisk = model.survival_score_cluster[0:max_num_cluster].cpu().tolist()

    for batch_id, sample in enumerate(train_data_loader, print_freq):
        # samples = samples.to(device)
        # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        sample = [element.to(device) for element in sample]
        (patientEmbedding, pos, keyPaddingMask, cluster, Dead, followUpTime,patientIdx) = sample
        event_indicator_train=np.append(event_indicator_train,Dead.cpu().numpy().astype(bool))
        event_time_train=np.append(event_time_train,followUpTime.cpu().numpy())  
        outputs = model(patientEmbedding, pos, keyPaddingMask, cluster)
        estimate_train = np.append(estimate_train,outputs[0].cpu().numpy())
        survival_score_patches = np.append(survival_score_patches,outputs[1].cpu().numpy().flatten())
        all_clusters = np.append(all_clusters,cluster.cpu().numpy().flatten())

    for batch_id, sample in enumerate(test_data_loader, print_freq):
        sample = [element.to(device) for element in sample]
        (patientEmbedding, pos, keyPaddingMask, cluster, Dead, followUpTime,patientIdx) = sample
        event_indicator_test=np.append(event_indicator_test,Dead.cpu().numpy().astype(bool))
        event_time_test=np.append(event_time_test,followUpTime.cpu().numpy())  
        outputs = model(patientEmbedding, pos, keyPaddingMask, cluster)
        estimate_test = np.append(estimate_test,outputs[0].cpu().numpy())
        
    CIndex = concordance_index_censored(event_indicator_test, event_time_test, estimate_test)
    breslow = BreslowEstimator().fit(estimate_train, event_indicator_train, event_time_train)
    sample_surv_fn_train = breslow.get_survival_function(estimate_train)
    sample_surv_fn_test = breslow.get_survival_function(estimate_test)
    ytr = np.array([(bool(e),t) for e,t in zip(event_indicator_train,event_time_train)], dtype=[('Dead', bool),('Follow-up Time',float)])
    ytest = np.array([(bool(e),t) for e,t in zip(event_indicator_test,event_time_test)], dtype=[('Dead', bool),('Follow-up Time',float)])
    IPCWCIndex = concordance_index_ipcw(ytr, ytest, estimate_test)
    min_time_tr = event_time_train.min()
    min_time_ts = event_time_test.min()
    max_time_tr = event_time_train.max()
    max_time_ts = event_time_test.max()

    _min = min_time_ts if min_time_ts >= min_time_tr else min_time_tr
    _max = max_time_ts if max_time_ts <= max_time_tr else max_time_tr
    time_points = np.arange(_min, _max, 0.01)

    # Integrated Brier Score
    preds_train = np.asarray([[fn(t) for t in time_points] for fn in sample_surv_fn_train])
    IBSTrain=integrated_brier_score(ytr, ytr, preds_train, time_points)

    preds_test = np.asarray([[fn(t) for t in time_points] for fn in sample_surv_fn_test])
    IBSTest=integrated_brier_score(ytr, ytest, preds_test, time_points)
    stats = {'CIndex':CIndex[0],'IPCWCIndex':IPCWCIndex[0],'IBSTrain':IBSTrain,'IBSTest':IBSTest}
    for i in range(0,max_num_cluster):
        meanClusterRisk.append(np.mean(survival_score_patches[all_clusters==i]))
        stdClusterRisk.append(np.std(survival_score_patches[all_clusters==i]))
    clusterRiskDataFrame = pd.DataFrame({'meanClusterRisk':meanClusterRisk,'stdClusterRisk':stdClusterRisk,'LearnedClusterRisk':LearnedClusterRisk})
    clusterRiskDataFrame.to_csv(output_dir/f'clusterRisk_fold{fold}.csv')
    return stats
