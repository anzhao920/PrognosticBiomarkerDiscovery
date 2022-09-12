import logging
from datetime import datetime
now = datetime.now()
dt_string = now.strftime("%d%m%Y_%H%M%S")
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    handlers=[
        logging.FileHandler(f'log_file_{dt_string}.log'),
        logging.StreamHandler()        
    ],
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.DEBUG)

log = logging.getLogger()

def get_lr(optimizer):
    lrs = []
    for param_group in optimizer.param_groups:
        lr = float(param_group['lr'])
        lrs.append(lr)

    return max(lrs)

class AverageMeter(object):
    """Computes and stores the average and current value""" 
    def __init__(self):
        self.reset()
                   
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0 

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count