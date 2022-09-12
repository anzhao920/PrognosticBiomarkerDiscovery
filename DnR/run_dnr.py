import os
import argparse
import io
from tqdm import tqdm
from torch.utils.data import DataLoader,random_split
from dnr import CAE_DNR, NonParametricClassifier, ANsDiscovery, Criterion
import torchvision
from torchvision import transforms
import torch.optim as optim
import torch
from utils import get_lr,log
from torch.utils.tensorboard import SummaryWriter
import logging
import numpy as np
import time
import webdataset as wds
import torch.nn as nn
def latentVariable_func(model, data_loader, save_folder, norm, projectionHead):

    model.eval()
    loss = None
    tqdm_iterator = tqdm(data_loader, desc='val')
    latentVariables = []
    indices = []
    for batch_idx, data in enumerate(tqdm_iterator):
        data = data[0] 
        data_in = data['image_he'].cuda().float()
        index = data['idx_overall']
        
        # calculate loss and metrics
        with torch.no_grad():
            latent_variable_batch = model.latent_variable(data_in,projectionHead)
            # latent_variable_batch = model.module.latent_variable(data_in,projectionHead)
            if norm:
                latent_variable_batch = torch.div(latent_variable_batch, torch.norm(latent_variable_batch+1e-12, p=2, dim=1, keepdim=True))
        if batch_idx==0:
            latentVariables = latent_variable_batch.cpu().detach().numpy()
            indices = index.cpu().detach().numpy()
        else:
            latentVariables = np.concatenate((latentVariables,latent_variable_batch.cpu().detach().numpy()),axis=0)
            indices = np.concatenate((indices,index.cpu().detach().numpy()),axis=0)
    np.save(save_folder+'latentVariables.npy',latentVariables)
    np.save(save_folder+'indices.npy',indices)
    return latentVariables, indices

def model_func(model, optimizer, data_loader, batch_num, npc, ANs_discovery, criterion, round, n_samples, epoch, tb_writer, save_folder,save_log_interval=100,save_checkpoint_epoch_interval=1):

    model.train()
    loss = None
    tqdm_iterator = tqdm(data_loader, desc='train')
    for batch_idx, data in enumerate(tqdm_iterator):
        data = data[0]        
        data_in = data['image_he'].cuda().float()
        data_out = data['image'].cuda().float()
        index = data['idx_overall'].cuda().long()

        if 'image_pairs' in data:
            data_in_p = data['image_pairs_he'].cuda().float()
            data_out_p = data['image_pairs'].cuda().float()
            index_p = data['idx_overall'].cuda().long() + n_samples

            data_in = torch.cat((data_in, data_in_p), 0)
            data_out = torch.cat((data_out, data_out_p), 0)
            index = torch.cat((index, index_p), 0)

        optimizer.zero_grad()
        x_hat, zp, zb = model(data_in,decode = True)
        # calculate loss and metrics
        res = criterion(data_out, index, npc, ANs_discovery, x_hat, zp)

        # Parse new loss and add to old one
        _loss = dict([(k, v.item()) for k, v in res.items()])
        loss = dict([(k, loss[k]+_loss[k]) for k in loss]) if loss is not None else _loss

        tqdm_iterator.set_postfix(dict([(k, v/(batch_idx+1)) for k, v in loss.items()]))
        # backward pass
        res['loss'].backward()
        # step
        current_lr = get_lr(optimizer)
        log.info(
                'Round: {}, Batch: [{}] [{}/{}], lr: {}, bach_loss_avg: '\
                .format(round, epoch, batch_idx, batch_num, current_lr)+tqdm_iterator.postfix)
        optimizer.step()   
        if batch_idx != 0 and (epoch*batch_num+batch_idx+1) % save_log_interval == 0:
            num=(epoch*batch_num+batch_idx+1)//save_log_interval
            temploss_interval=dict([(k, v/(batch_idx+1)) for k, v in loss.items()])
            tb_writer.add_scalar('train/loss_interval'+'Round{}'.format(round), temploss_interval['loss'],num)
            tb_writer.add_scalar('train/loss_inst_interval'+'Round{}'.format(round), temploss_interval['loss_inst'], num)
            tb_writer.add_scalar('train/loss_ans_interval'+'Round{}'.format(round), temploss_interval['loss_ans'], num)
            tb_writer.add_scalar('train/loss_mse_interval'+'Round{}'.format(round), temploss_interval['loss_mse'], num)

    log.info(
        'Round: {}, Epoch: {}, epoch_loss: '.format(round, epoch)+tqdm_iterator.postfix)
    temploss = dict([(k, v/batch_num) for k, v in loss.items()])
    tb_writer.add_scalar('train/loss'+'Round{}'.format(round), temploss['loss'], epoch)
    tb_writer.add_scalar('train/loss_inst'+'Round{}'.format(round), temploss['loss_inst'], epoch)
    tb_writer.add_scalar('train/loss_ans'+'Round{}'.format(round), temploss['loss_ans'], epoch)
    tb_writer.add_scalar('train/loss_mse'+'Round{}'.format(round), temploss['loss_mse'], epoch)
    if (epoch+1) % save_checkpoint_epoch_interval == 0:
        model_save_path = '{}_round_{}_epoch_{}.pth.tar'.format(save_folder, round, epoch)
        model_save_dir = os.path.dirname(model_save_path)
        if not os.path.exists(model_save_dir):
            os.makedirs(model_save_dir)
        
        log.info('Save checkpoints: Round = {} epoch = {}'.format(round, epoch)) 
        torch.save({
                    'round':round,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'model_npc_state_dict': npc.state_dict(),
                    'model_ans_state_dict': ANs_discovery.state_dict(),
                    'optimizer': optimizer.state_dict()
                    },
                    model_save_path)

def identity(x):
    return x

def transform(x):
    x = x.item()
    keys = ['image_he','image','image_pairs_he','image_pairs']
    for key in keys:
        data = x[key]
        if not torch.is_tensor(data):
            x[key] = transforms.functional.to_tensor(data)           
    return x    
    
def npy_allow_pickle_decoder(value):
    import numpy.lib.format
    stream = io.BytesIO(value)
    return numpy.lib.format.read_array(stream,allow_pickle=True)

def main():

    parser = argparse.ArgumentParser(description='Run DnR')
    parser.add_argument('--output', dest='output', type=str,
                        default='./trainedModels/', help='Output path')
    parser.add_argument('--db', dest='db', type=str,
                        default='./Shards/', help='Path to database')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--n_channels', default=2, type=int)
    parser.add_argument('--max_round', default=4, type=int)
    parser.add_argument('--max_epoch', default=25, type=int)
    parser.add_argument('--name', type=str,default='resnet18', help='backbone')    
    parser.add_argument('--len_allDataset', default=1547467, type=int, help='number of training samples')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--phase', default='train',type=str,help='train or test')
    parser.add_argument('--trained_model', type=str,
                        default='./trainedModels/GPU/ResNet18_25rounds/DataParallel_model_3_24.pth/', help='Path to trained models')

    args = parser.parse_args()

    name=args.name
    data_train_dir=list(args.db+'IPFCTDatasetDnR64-{:06d}.tar'.format(i) for i in range(0,155))
    drop_last = False
    pin_memory = True

    if name =='resnet18' or name =='resnet34':
        hidden_dimension =512
        npc_dimension = 128
    elif name =='resnet50':
        hidden_dimension = 2048
        npc_dimension = 512
    tb_writer = SummaryWriter()
    
    batch_num = args.len_allDataset//args.batch_size
    ds_train = (
        wds.WebDataset(data_train_dir)
        .shuffle(5000)
        .decode(wds.handle_extension(".npy", npy_allow_pickle_decoder))
        .to_tuple("npy","metadata.pyd")
        .map_tuple(transform,identity)
    )

    dl_train = wds.WebLoader(
            dataset=ds_train,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            drop_last=drop_last,
            pin_memory=pin_memory
        )

    log.info('Build model with n_channels: {} ...'.format(args.n_channels))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.phase == 'train':       
        model = nn.DataParallel(CAE_DNR(pretrained=True, n_channels=args.n_channels, hidden_dimension=hidden_dimension, name=name, npc_dimension = npc_dimension)).to(device)
        npc = NonParametricClassifier(npc_dimension, 2*args.len_allDataset).to(device)
        ANs_discovery = ANsDiscovery(2*args.len_allDataset).to(device)
        criterion = Criterion()
        optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999))
        model_save = os.path.join(args.output, '{}_model'.format(model.__class__.__name__))
        start_epoch = 0  # start from epoch 0 or last checkpoint epoch
        start_round = 0  # start for iter 0 or last checkpoint iter
        round = start_round

        # At each round we increase the entropy threshold to select NN
        while round < args.max_round:

            # variables are initialized to different value in the first round
            is_first_round = True if round == start_round else False

            if not is_first_round:
                ANs_discovery.update(round, npc, None)

            # start to train for an epoch
            epoch = start_epoch if is_first_round else 0
            while epoch < args.max_epoch:
                log.info('Round: {}/{}, epoch: {}/{}'.format(round, args.max_round, epoch, args.max_epoch))

                # 1. Train model (1 epoch)
                model_func(model=model, optimizer=optimizer, data_loader=dl_train,
                        batch_num=batch_num,npc=npc, ANs_discovery=ANs_discovery, criterion=criterion,
                        round=round, n_samples=args.len_allDataset,epoch=epoch,tb_writer=tb_writer,save_folder = args.output)
                # if epoch != 0 and (epoch+1) % 5 == 0:
                #     torch.save(model, model_save+"_{}_{}.pth".format(round, epoch))
                #     torch.save(npc, model_save+"_npc_{}_{}.pth".format(round, epoch))
                #     torch.save(ANs_discovery, model_save+"_ans_{}_{}.pth".format(round, epoch))

                epoch += 1

            # log best accuracy after each iteration
            round += 1
        tb_writer.flush()
        tb_writer.close()
    else:
        model=torch.load(args.trained_model)
        latentVariable_func(model=model, data_loader=dl_train, save_folder = args.output, norm=True, projectionHead=True)


if __name__ == '__main__':
    main()

