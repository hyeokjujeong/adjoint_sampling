# Copyright (c) Meta Platforms, Inc. and affiliates.

import os
from contextlib import contextmanager
from typing import Optional

import torch

from fairchem.core import OCPCalculator

from huggingface_hub import hf_hub_download

from adjoint_sampling.energies.regularizers import bond_structure_regularizer


@contextmanager
def wandb_mode_disabled():
    """
    Temporarily sets the WANDB_MODE environment variable to 'disabled'.
    Saves the original value of WANDB_MODE, sets it to 'disabled' within the context,
    and restores the original value when exiting the context.
    """
    original_mode = os.environ.get("WANDB_MODE")
    try:
        os.environ["WANDB_MODE"] = "disabled"
        yield
    finally:
        if original_mode is None:
            del os.environ["WANDB_MODE"]
        else:
            os.environ["WANDB_MODE"] = original_mode


class FairChemEnergy(torch.nn.Module):
    def __init__(
        self,
        model_ckpt: Optional[str] = None,
        tau: float = 1e-3,
        alpha: float = 1e3,
        device: str = "cpu",
        default_regularize: bool = True,
    ):
        super().__init__()
        if model_ckpt is None:
            model_ckpt = hf_hub_download(
                repo_id="facebook/adjoint_sampling", filename="esen_spice.pt"
            )
        with wandb_mode_disabled():
            calculator = OCPCalculator(
                checkpoint_path=model_ckpt, cpu=(device == "cpu"), seed=0
            )
        self.predictor = calculator.trainer
        m = self.predictor.model
        inner = m.module if isinstance(m, torch.nn.parallel.DistributedDataParallel) else m

        inner.backbone.use_pbc = True
        inner.backbone.use_pbc_single = False

        self.device = device
        self.tau = tau
        self.alpha = alpha
        self.r_max = inner.backbone.cutoff
        self.atomic_numbers = torch.arange(100)
        self.default_regularize = default_regularize

    def bond_regularizer(self, batch):
        energy_reg = bond_structure_regularizer(
            batch["positions"],
            batch["edge_attrs"][:, 0].unsqueeze(-1),  # bond limits
            batch["edge_attrs"][:, 1].unsqueeze(-1),  # bond types
            batch["edge_index"],
            batch["ptr"],
            alpha=self.alpha,
        )
        grad_outputs = [torch.ones_like(energy_reg)]
        gradient = torch.autograd.grad(
            outputs=[energy_reg],  # [n_graphs, ]
            inputs=[batch["positions"]],  # [n_nodes, 3]
            grad_outputs=grad_outputs,
            retain_graph=False,  # Make sure the graph is not destroyed during training
            create_graph=False,  # Create graph for second derivative
            allow_unused=True,  # For complete dissociation turn to true
        )[0]
        return energy_reg, -gradient

    def __call__(self, batch, regularize: Optional[bool] = None):
        if regularize is None:
            regularize = self.default_regularize

        # rename/add required input fields for fairchem model
        batch.natoms = batch.ptr[1:] - batch.ptr[:-1]
        batch.atomic_numbers = batch["node_attrs"].argmax(dim=-1)

        # TODO use fake large cell
        batch.cell = (
            batch.cell.view(-1, 3, 3)
            + torch.eye(3).to(batch.cell.device).unsqueeze(0) * 1e3
        ).float()
        batch.pos = batch.positions.float()  # wrap?
        # note that our model has otf graph. edge_index here is not used.

        output_dict = {}
        preds = self.predictor._forward(batch)
        for target_key in self.predictor.config["outputs"]:
            output_dict[target_key] = self.predictor._denorm_preds(
                target_key, preds[target_key], batch
            )

        if regularize:
            reg_energy, reg_force = self.bond_regularizer(batch)
            output_dict["reg_forces"] = reg_force.detach()  # / self.tau
            output_dict["reg_energy"] = reg_energy.detach()
        else:
            output_dict["reg_forces"] = torch.zeros_like(output_dict["forces"])
            output_dict["reg_energy"] = torch.zeros_like(output_dict["energy"])

        output_dict["forces"] = (output_dict["forces"].detach()) / self.tau
        return output_dict
