# Copyright (c) Meta Platforms, Inc. and affiliates.

import builtins
import datetime
import os
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from omegaconf import open_dict


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print("[{}] ".format(now), end="")  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def get_shared_folder(shared_dir: str) -> Path:
    user = os.getenv("USER")
    if Path(shared_dir).is_dir():
        p = Path(shared_dir) / user / "distributed"
        p.mkdir(exist_ok=True)
        return p
    raise RuntimeError("No shared folder available")


def get_init_file(shared_dir: str):
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder(shared_dir)), exist_ok=True)
    init_file = get_shared_folder(shared_dir) / f"{os.environ['SLURM_JOBID']}_init"
    return init_file


def init_distributed_mode(cfg):

    with open_dict(cfg):
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            cfg.rank = int(os.environ["RANK"])
            cfg.world_size = int(os.environ["WORLD_SIZE"])
            cfg.gpu = int(os.environ["LOCAL_RANK"])
            cfg.dist_url = "env://"
        elif (
            "SLURM_PROCID" in os.environ and os.environ["SLURM_JOB_NAME"] != "bash"
        ):  # Exclude interactive shells
            cfg.rank = int(os.environ["SLURM_PROCID"])
            cfg.world_size = int(os.environ["SLURM_GPUS_PER_NODE"]) * int(
                os.environ["SLURM_JOB_NUM_NODES"]
            )
            cfg.gpu = cfg.rank % torch.cuda.device_count()
            cfg.dist_url = get_init_file(cfg.shared_dir).as_uri()
        else:
            print("Not using distributed mode")
            cfg.distributed = False
            return

        cfg.distributed = True

        torch.cuda.set_device(cfg.gpu)
        cfg.dist_backend = "nccl"
        print(
            "| distributed init (rank {}): {}, gpu {}".format(
                cfg.rank, cfg.dist_url, cfg.gpu
            ),
            flush=True,
        )
        torch.distributed.init_process_group(
            backend=cfg.dist_backend,
            init_method=cfg.dist_url,
            world_size=cfg.world_size,
            rank=cfg.rank,
            timeout=timedelta(seconds=240),
        )
        torch.distributed.barrier()
        setup_for_distributed(cfg.rank == 0)
