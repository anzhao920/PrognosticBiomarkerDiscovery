import argparse
import datetime
import json
import random
from re import A
import time
from pathlib import Path
import os

from torch.utils.data.dataset import Subset
from datasets.MyData import MyDataset
import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler,random_split
from sklearn.model_selection import KFold
import util.misc as utils

from models.engine import evaluate, train_one_epoch_SAM,test
from models import build_model
from models.sam import SAM
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    parser.add_argument('--position_embedding', default='3Dlearned', type=str,
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true',default=False)
    parser.add_argument('--pretrained_path', default='', type=str, help="path of pretrained model")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')


    parser.add_argument('--output_dir', default='./',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--kfoldNum', default=5, type=int)
    parser.add_argument('--dataDir',type=str,help='path of the data')
    parser.add_argument('--externalDataDir',type=str,help='path of the external test data')
    


    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    # cluster parameters
    parser.add_argument('--group_Q', action='store_true', default=False)
    parser.add_argument('--group_K', action='store_true', default=False)
    parser.add_argument('--cuda-devices', default=None)
    parser.add_argument('--max_num_cluster', default=64, type=int)
    parser.add_argument('--sequence_len', default=15000, type=int)

    # gridsearch parameters
    parser.add_argument('--withPosEmbedding', action='store_true', default=False)
    parser.add_argument('--seq_pool', action='store_true', default=False,help='use attention pooling layer for aggregating patch risk score') 
    parser.add_argument('--withLN', action='store_true', default=False)
    parser.add_argument('--withEmbeddingPreNorm', action='store_true', default=False,help='Pre-normalize the patch representation before feeding them into ViT')
    parser.add_argument('--input_pool', action='store_true', default=False)
    parser.add_argument('--mixUp', action='store_true', default=False)
    parser.add_argument('--SAM', action='store_true', default=False)


    return parser

def main(args):
    allDataset = MyDataset(root_dir=args.dataDir,sequence_len=args.sequence_len,max_num_cluster=args.max_num_cluster,status = 'test',input_pool=args.input_pool)
    kfoldSplits=KFold(n_splits=args.kfoldNum,shuffle=True,random_state=args.seed)  
    splitIdx = kfoldSplits.split(np.arange(len(allDataset)))
    CIndexTest = []
    CIndexExternalTest = []
    IBSTest = []
    IBSExternalTest = []
    correlationCoeffTest = []
    IPCWCIndexTest = []
    IPCWCIndexExternalTest = []
    for fold, (train_idx,test_idx) in enumerate(splitIdx):
        utils.init_distributed_mode(args)
        print(args)
        device = torch.device(args.device)

        # fix the seed for reproducibility
        seed = args.seed + utils.get_rank()
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        model, criterion = build_model(args)
        model.to(device)

        model_without_ddp = model
        if args.distributed:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            model_without_ddp = model.module
        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)
        if args.pretrained_path!='':
            pretrainedModel = torch.load(args.pretrained_path)
            model_without_ddp.load_state_dict(pretrainedModel['model'], strict=False)

        if args.SAM:
            base_optimizer = torch.optim.Adam # define an optimizer for the "sharpness-aware" update
            optimizer_SAM = SAM(model_without_ddp.parameters(), base_optimizer,lr=args.lr,weight_decay=args.weight_decay)
            optimizer = optimizer_SAM
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop) 
        else:
            optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr,
                                          weight_decay=args.weight_decay)
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

        best_loss = 1e12
        tb_writer = SummaryWriter()
        dataset_train,dataset_val = random_split(Subset(allDataset,train_idx),[int(len(train_idx)*0.8),len(train_idx)-int(len(train_idx)*0.8)],generator=torch.Generator().manual_seed(args.seed))
        dataset_test = Subset(allDataset,test_idx)
        dataset_train_all = Subset(allDataset,train_idx)
        
        if args.distributed:
            sampler_train = DistributedSampler(dataset_train)
            sampler_val = DistributedSampler(dataset_val, shuffle=False)
            sampler_test = DistributedSampler(dataset_test, shuffle=False)
        else:
            sampler_train = torch.utils.data.RandomSampler(dataset_train)
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            sampler_test = torch.utils.data.SequentialSampler(dataset_test)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, args.batch_size, drop_last=False)

        
        data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                    collate_fn=None, num_workers=args.num_workers)
        data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                    drop_last=False,num_workers=args.num_workers)
        data_loader_test = DataLoader(dataset_test, args.batch_size, sampler=sampler_test,
                                    drop_last=False, num_workers=args.num_workers)
        
        output_dir = Path(args.output_dir)
        if args.kfoldNum>1:
            output_dir = output_dir/f'fold{fold}'
            output_dir.mkdir(parents=True, exist_ok=True)
            
        if args.resume:
            checkpoint = torch.load(args.resume, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])
            if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer'])
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                args.start_epoch = checkpoint['epoch'] + 1      

        if args.eval:
            val_stats = evaluate(model, criterion, data_loader_train, device, args.output_dir,'Validation')

        print(f"fold {fold}/{args.kfoldNum} Start training")
        start_time = time.time()
        # train the model 
        for epoch in range(args.start_epoch, args.epochs):
            if args.distributed:
                sampler_train.set_epoch(epoch)

            train_stats = train_one_epoch_SAM(
                model, criterion, data_loader_train, optimizer, device, epoch,fold,tb_writer,
                args.clip_max_norm,mixUp=args.mixUp,SAM=args.SAM)    

            lr_scheduler.step()

            train_eval_stats = evaluate(
                model, criterion, data_loader_train, device, args.output_dir,'Train')
            val_stats = evaluate(
                model, criterion, data_loader_val, device, args.output_dir, 'Validation')
            test_stats = evaluate(
                model, criterion, data_loader_test, device, args.output_dir, 'Test')

            is_best = val_stats['loss'] < best_loss
            best_loss = min(val_stats['loss'], best_loss)            

            if args.output_dir:
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                # extra checkpoint before LR drop and every 100 epochs
                if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                if is_best:
                    checkpoint_paths.append(output_dir / f'model_best.pth.tar')
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model_without_ddp.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'args': args,
                    }, checkpoint_path)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'train_eval{k}': v for k, v in train_eval_stats.items()},
                        **{f'val_{k}': v for k, v in val_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}

            temploss = train_eval_stats
            tb_writer.add_scalar('train/loss'+'fold{}'.format(fold), temploss['loss'], epoch)
            tb_writer.add_scalar('train/CIndex'+'fold{}'.format(fold), temploss['CIndex'], epoch)                        
            temploss = val_stats
            tb_writer.add_scalar('val/loss'+'fold{}'.format(fold), temploss['loss'], epoch)
            tb_writer.add_scalar('val/CIndex'+'fold{}'.format(fold), temploss['CIndex'], epoch) 
            temploss = test_stats
            tb_writer.add_scalar('test/loss'+'fold{}'.format(fold), temploss['loss'], epoch)
            tb_writer.add_scalar('test/CIndex'+'fold{}'.format(fold), temploss['CIndex'], epoch)

            if args.output_dir and utils.is_main_process():
                with (output_dir / f"trainingLog_fold{fold}.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")  
        # evaluate the best model on internal and external datasets
        bestModelPath =  output_dir / f'model_best.pth.tar' 
        bestCheckpoint = torch.load(bestModelPath)
        bestModel, testcriterion = build_model(args)
        bestModel.to(device)
        bestModel.load_state_dict(bestCheckpoint['model'])
        data_loader_train_all = DataLoader(dataset_train_all, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        dataset_external_test = MyDataset(root_dir=externalDataDir,sequence_len=args.sequence_len,max_num_cluster=args.max_num_cluster,status='externalTest',input_pool=args.input_pool)
        dataset_external_test.status = 'externalTest'
        data_loader_external_test = DataLoader(dataset_external_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        externalOutputDir = output_dir / 'externalTest'
        Path(externalOutputDir).mkdir(parents=True, exist_ok=True)
        internalTestBestModelStatus = test(bestModel, testcriterion, data_loader_test, data_loader_train_all, device, output_dir,fold,coxBiomarkerRisk)
        externalTestBestModelStatus = test(bestModel, testcriterion, data_loader_external_test, data_loader_train_all, device, externalOutputDir,fold)
        log_stats = {
            **{f'testBestModel_{k}': v for k, v in internalTestBestModelStatus.items()},
            **{f'ExternalTestBestModel_{k}': v for k, v in externalTestBestModelStatus.items()},
            'fold': fold
            }

        CIndexTest.append(internalTestBestModelStatus['CIndex'])
        CIndexExternalTest.append(externalTestBestModelStatus['CIndex'])
        IPCWCIndexTest.append(internalTestBestModelStatus['IPCWCIndex'])
        IPCWCIndexExternalTest.append(externalTestBestModelStatus['IPCWCIndex'])
        IBSTest.append(internalTestBestModelStatus['IBSTest'])
        IBSExternalTest.append(externalTestBestModelStatus['IBSTest'])

        if args.output_dir and utils.is_main_process():
            with (output_dir / f"testingLog_fold{fold}.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")  
                
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    output_dir = Path(args.output_dir)
    AverageBestModelStatus = {'AverageCIndexTest':np.mean(CIndexTest), 'AverageCIndexExternalTest':np.mean(CIndexExternalTest),\
        'AverageIPCWCIndexTest':np.mean(IPCWCIndexTest), 'AverageIPCWCIndexExternalTest':np.mean(IPCWCIndexExternalTest),\
        'AverageIBSTest':np.mean(IBSTest),'AverageIBSExternalTest':np.mean(IBSExternalTest),\
        'stdCIndexTest':np.std(CIndexTest), 'stdCIndexExternalTest':np.std(CIndexExternalTest),\
        'stdIPCWCIndexTest':np.std(IPCWCIndexTest), 'stdIPCWCIndexExternalTest':np.std(IPCWCIndexExternalTest),\
        'stdIBSTest':np.std(IBSTest),'stdIBSExternalTest':np.std(IBSExternalTest),'AverageCorrelationCoeffTest':np.mean(correlationCoeffTest)}
    log_stats = {
        **{f'{k}': v for k, v in AverageBestModelStatus.items()}
        }    
    if args.output_dir and utils.is_main_process():
        with (output_dir / "logAverage.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")      
    print(AverageBestModelStatus)

if __name__ == '__main__':
    now = datetime.datetime.now()
    dt_string = now.strftime("%d%m%Y_%H%M%S")    
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    if args.cuda_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda_devices
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    hyperparameters = vars(args)
    hyperparameter_stas = {
        **{f'{k}': v for k, v in hyperparameters.items()}
        }
    if args.output_dir and utils.is_main_process():
        with (Path(args.output_dir) / "logHyperparameters.txt").open("a") as f:
            f.write(json.dumps(hyperparameter_stas) + "\n")      
    main(args)
