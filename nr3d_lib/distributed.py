"""
@file   distributed.py
@author Chenjing Ding, Sensetime & Jianfei Guo, Shanghai AI Lab
@brief  Utilities for distributed models & training.
"""

import os
import torch
import random
import numpy as np
from typing import Optional

import torch.distributed as dist
import torch.multiprocessing as mp

rank = 0        # process id, for IPC
local_rank = 0  # local GPU device id
world_size = 1  # number of processes

def init_env(args, seed=42):
    global rank, local_rank, world_size
    if args.ddp:
        msg = f"=> Enter init_process_group(): "
        if 'RANK' in os.environ:
            msg += f"\n\t=> rank={int(os.environ['RANK'])}"
        if 'WORLD_SIZE' in os.environ:
            msg += f"\n\t=> world_size={int(os.environ['WORLD_SIZE'])}"
        if 'LOCAL_RANK' in os.environ:
            msg += f"\n\t=> local_rank={int(os.environ['LOCAL_RANK'])}"
        if 'MASTER_ADDR' in os.environ:
            msg += f"\n\t=> master_addr={os.environ['MASTER_ADDR']}"
        if 'MASTER_PORT' in os.environ:
            msg += f"\n\t=> master_port={os.environ['MASTER_PORT']}"
        print(msg)
        
        #------------- multi process running, using DDP
        dist_launcher: str = args.get('dist_launcher', 'pytorch').lower()
        if ('SLURM_PROCID' in os.environ) or dist_launcher == 'slurm':
            #--------- For SLURM
            slurm_initialize('nccl', port=args.port)
        elif dist_launcher == 'aliyun':
            #--------- For Aliyun PAI
            aliyun_pai_initialize(backend='nccl', port=args.port)
        else:
            #--------- For torch.distributed.launch
            dist.init_process_group(backend='nccl')

        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(local_rank)
        args.device_ids = [local_rank]
        msg = f"=> Done init Env @ DDP: "
        msg += f"\n\t=> device_ids set to {args.device_ids}"
        if 'RANK' in os.environ:
            msg += f"\n\t=> rank={int(os.environ['RANK'])}"
        if 'WORLD_SIZE' in os.environ:
            msg += f"\n\t=> world_size={int(os.environ['WORLD_SIZE'])}"
        if 'LOCAL_RANK' in os.environ:
            msg += f"\n\t=> local_rank={int(os.environ['LOCAL_RANK'])}"
        if 'MASTER_ADDR' in os.environ:
            msg += f"\n\t=> master_addr={os.environ['MASTER_ADDR']}"
        if 'MASTER_PORT' in os.environ:
            msg += f"\n\t=> master_port={os.environ['MASTER_PORT']}"
        print(msg)
    else:
        #------------- single process running, using single GPU or DataParallel
        # torch.cuda.set_device(args.device_ids[0])
        print(f"=> Init Env @ single process: use device_ids = {args.device_ids}")
        rank = 0
        local_rank = args.device_ids[0]
        world_size = 1
        torch.cuda.set_device(args.device_ids[0])
    set_seed(seed)


def slurm_initialize(backend='nccl', port: Optional[int] = None):
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    if '[' in node_list:
        beg = node_list.find('[')
        pos1 = node_list.find('-', beg)
        if pos1 < 0:
            pos1 = 1000
        pos2 = node_list.find(',', beg)
        if pos2 < 0:
            pos2 = 1000
        node_list = node_list[:min(pos1, pos2)].replace('[', '')
    addr = node_list[8:].replace('-', '.')
    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' not in os.environ:
        os.environ["MASTER_PORT"] = "13333"
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    if backend == 'nccl':
        dist.init_process_group(backend='nccl')
    else:
        dist.init_process_group(backend='gloo', rank=proc_id, world_size=ntasks)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    os.environ['LOCAL_RANK'] = str(device)

def aliyun_pai_initialize(backend='nccl', port: Optional[int] = None):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    if port is not None:
        os.environ['MASTER_PORT'] = str(port)
    elif 'MASTER_PORT' not in os.environ:
        os.environ["MASTER_PORT"] = "13333"

    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    os.environ['LOCAL_RANK'] = str(device)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_master():
    return rank == 0

def get_rank():
    return int(os.environ.get('SLURM_PROCID', rank))

def get_local_rank():
    return int(os.environ.get('LOCAL_RANK', local_rank))

def get_world_size():
    return int(os.environ.get('SLURM_NTASKS', world_size))
