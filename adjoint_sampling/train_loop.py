# Copyright (c) Meta Platforms, Inc. and affiliates.

import torch
from torchmetrics.aggregation import MeanMetric
import torch.distributed as dist
from adjoint_sampling.components.soc import (
    adjoint_score_target,
    adjoint_score_target_torsion,
)
from adjoint_sampling.sampletorsion.torsion import check_torsions

from adjoint_sampling.utils.data_utils import cycle


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train_one_epoch(
    controller,
    noise_schedule,
    clipper,
    train_dataloader,
    optimizer,
    warmup_scheduler,
    lr_schedule,
    device,
    cfg,
    pretrain_mode=False,
):
    epoch_loss = 0
    controller.train(True)
    #print("controller in train mode:", flush=True)
    epoch_loss = MeanMetric().to(device, non_blocking=True)
    loader = iter(train_dataloader)
    #print(f"[rank {dist.get_rank()}] loader initialized", flush=True)
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    #print("loader initialized in train loop", flush=True)
    skipped = 0
    for i in range(cfg.num_batches_per_epoch):
        # print(get_lr(optimizer))
        optimizer.zero_grad()
        #if dist.is_available() and dist.is_initialized():
        #    print(f"[rank {dist.get_rank()}] before next(i={i})", flush=True)
        try:
            graph_state_1, grad_E = next(loader)
        except StopIteration:
            loader = iter(train_dataloader)
            graph_state_1, grad_E = next(loader)
        #if dist.is_available() and dist.is_initialized():
        #    print(f"[rank {dist.get_rank()}] after next(i={i})", flush=True)
        n_systems = len(graph_state_1["ptr"]) - 1
        nonfinite = (not torch.isfinite(graph_state_1["positions"]).all().item())

        if dist.is_available() and dist.is_initialized():
            flag = torch.tensor([int(n_systems == 0 or nonfinite)], device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.SUM)
            if flag.item() > 0:
                skipped += 1
                if dist.get_rank() == 0:
                    print("[global] skip step due to empty/nonfinite batch", flush=True)
                # 모두가 '같이' 건너뛰기
                continue
        else:
            if (n_systems == 0) or nonfinite:
                print("[single] skip step due to empty/nonfinite batch", flush=True)
                continue
        #print("loader loaded in train loop", flush=True)
        graph_state_1 = graph_state_1.to(device)
        n_systems = len(graph_state_1["ptr"]) - 1
        t = torch.rand(n_systems).to(device)
        if cfg.learn_torsions:
            check_torsions(
                graph_state_1["positions"],
                graph_state_1["tor_index"],
                graph_state_1["torsions"],
            )
        graph_state_t = noise_schedule.sample_posterior(t, graph_state_1)
        #print(graph_state_t)\
        #print("graph_state_t sampled in train loop", flush=True)
        predicted_score = controller(t, graph_state_t) 
        #print("predicted score calculated in train loop", flush=True)

        if cfg.learn_torsions:
            g_t = torch.repeat_interleave(
                noise_schedule.g(t), graph_state_1["n_torsions"]
            )
            alpha_t = torch.repeat_interleave(
                noise_schedule.alpha(t), graph_state_1["n_torsions"]
            )
            score_target = adjoint_score_target_torsion(grad_E, clipper)
        else:
            g_t = noise_schedule.g(t)[graph_state_1["batch"], None]
            alpha_t = noise_schedule.alpha(t)[graph_state_1["batch"], None]
            score_target = adjoint_score_target(
                graph_state_1, grad_E, noise_schedule, clipper, no_pbase=cfg.no_pbase
            )
        #print("score target calculated in train loop", flush=True)
        if cfg.use_AM_SDE:
            predicted_score = predicted_score / g_t

        adjoint_loss = (predicted_score - score_target).pow(2).sum(-1).mean(0)
        #print("adjoint loss calculated in train loop", flush=True)
        if cfg.scaled_BM_loss and cfg.learn_torsions:
            bm_loss = (
                (
                    predicted_score * alpha_t.pow(2)
                    - (graph_state_1["torsions"] - graph_state_t["torsions"])
                )
                .pow(2)
                .mean(0)
            )
        elif cfg.scaled_BM_loss and not cfg.learn_torsions:
            bm_loss = (
                (
                    predicted_score * alpha_t.pow(2)
                    - (graph_state_1["positions"] - graph_state_t["positions"])
                )
                .pow(2)
                .sum(-1)
                .mean(0)
            )
        elif not cfg.scaled_BM_loss and cfg.learn_torsions:
            bm_loss = (
                (
                    predicted_score
                    - 1
                    / alpha_t.pow(2)
                    * (graph_state_1["torsions"] - graph_state_t["torsions"])
                )
                .pow(2)
                .mean(0)
            )
        else:  # not cfg.scaled_BM_loss and not cfg.learn_torsions:
            bm_loss = (
                (
                    predicted_score
                    - 1
                    / alpha_t.pow(2)
                    * (graph_state_1["positions"] - graph_state_t["positions"])
                )
                .pow(2)
                .sum(-1)
                .mean(0)
            )

        if pretrain_mode and cfg.BM_only_pretrain:
            loss = bm_loss
        else:
            loss = adjoint_loss + cfg.BM_loss_weight * bm_loss
        #print("total loss calculated in train loop", flush=True)

        #if i == 0:
        #    for n, p in controller.named_parameters():
        #        p.register_hook(lambda g, n=n: print(f"[hook] grad arrived: {n}", flush=True))
        loss.backward()
        #print("loss backward done in train loop", flush=True)
        torch.nn.utils.clip_grad_norm_(controller.parameters(), cfg.grad_clip)
        #print("grad norm clipped in train loop", flush=True)
        optimizer.step()
        #print("optimizer step done in train loop", flush=True)
        epoch_loss.update(loss.item())
        #print("epoch loss updated in train loop", flush=True)
        with warmup_scheduler.dampening():
            if lr_schedule:
                lr_schedule.step()
        #print(f"[rank {dist.get_rank()}] skipped_steps={skipped}/{cfg.num_batches_per_epoch}", flush=True)
    return {"loss": float(epoch_loss.compute().detach().cpu())}
