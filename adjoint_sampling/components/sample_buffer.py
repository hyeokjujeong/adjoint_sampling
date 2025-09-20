# Copyright (c) Meta Platforms, Inc. and affiliates.

from torch.utils.data import DataLoader

from adjoint_sampling.utils.data_utils import PreBatchedDataset
import torch.utils.data.distributed 
import torch.distributed as dist

class BatchBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.batch_list = []

    def add(self, graph_states, grads):
        batch = [(graph_state, grad) for graph_state, grad in zip(graph_states, grads)]

        self.batch_list.extend(batch)

        if len(self.batch_list) > self.buffer_size:
            self.batch_list = self.batch_list[-self.buffer_size :]

    def get_data_loader(self, distributed=False, shuffle=True, drop_last=True, num_workers=0):
        dataset = PreBatchedDataset(self.batch_list)
        if distributed:
            ws = dist.get_world_size() if dist.is_initialized() else 1
            eff_drop_last = drop_last
            if len(dataset) < ws:
                eff_drop_last = False
            sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle, drop_last=eff_drop_last)
            dataloader = DataLoader(dataset, batch_size=None, sampler=sampler, num_workers=num_workers)
        else:
            dataloader = DataLoader(dataset, batch_size=None, shuffle=shuffle, num_workers=num_workers)
        return dataloader

    def save_state(self, filename):
        # TODO.
        pass

    def load_state(self, filename):
        # TODO.
        pass
