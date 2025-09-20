# Copyright (c) Meta Platforms, Inc. and affiliates.

import csv

import torch
from hydra.utils import to_absolute_path

from adjoint_sampling.components.datasets import xyz_to_loader

from adjoint_sampling.components.sde import integrate_sde
from adjoint_sampling.components.soc import SOC_loss, SOC_loss_torsion

from adjoint_sampling.utils.eval_utils import eval_batch
from adjoint_sampling.utils.visualize_utils import (
    get_dataset_fig,
    visualize_conformations,
)

ATOMIC_NUMBERS = {
    "H": 1,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Br": 35,
    "I": 53,
}
SYM_LIST = {v: k for k, v in ATOMIC_NUMBERS.items()}


def save_to_xyz(batches, outputs, atomic_numbers, rank, dir):
    k = 0
    for graph_state, output in zip(batches, outputs):
        positions = graph_state["positions"].detach().cpu()
        atom_list = graph_state["node_attrs"].detach().cpu()
        ij = 0
        for i in range(len(graph_state["ptr"]) - 1):
            sys_size = graph_state["ptr"][i + 1] - graph_state["ptr"][i]
            with open(dir + "/sample_{}_{}.xyz".format(rank, k), "w") as f:
                f.write("{}\n".format(sys_size.item()))
                f.write("   {}\n".format(output["energy"][i]))
                for j in range(sys_size):
                    atomic_number = atomic_numbers[torch.nonzero(atom_list[ij])]
                    symbol = SYM_LIST[atomic_number[0, 0].item()]
                    f.write(
                        "{0:s}    {1:>f}    {2:>f}    {3:>f}\n".format(
                            symbol,
                            positions[ij, 0].item(),
                            positions[ij, 1].item(),
                            positions[ij, 2].item(),
                        )
                    )
                    ij += 1
            k += 1


def evaluation(
    sde,
    energy_model,
    eval_sample_loader,
    noise_schedule,
    atomic_number_table,
    rank,
    device,
    cfg,
):
    states = []
    outputs = []
    soc_loss = 0.0
    for batch in eval_sample_loader:
        batch = batch.to(device)
        graph_state, controls = integrate_sde(
            sde, batch, cfg.eval_nfe, only_final_state=False
        )
        # print(energy_model(graph_state))
        #
        #energy_model.eval()
        output = energy_model(graph_state)

        # torch.save(graph_state, "samples.pt")
        # torch.save(outputs, "sample_outputs.pt")

        # generated_energies = outputs["energy"]
        states.append(graph_state)
        outputs.append(output)
        if cfg.learn_torsions:
            soc_loss += SOC_loss_torsion(
                controls, graph_state, output["energy"], noise_schedule
            )
        else:
            soc_loss += SOC_loss(
                controls, graph_state, output["energy"], noise_schedule
            )

    # if not os.path.exists(cfg.sample_dir):
    #    os.makedirs(cfg.sample_dir)
    # save_to_xyz(states, outputs, atomic_number_table, rank, cfg.sample_dir)
    # soc_loss = SOC_loss(controls, states[0], outputs[0]["energy"], noise_schedule)
    soc_loss = (soc_loss / cfg.num_eval_samples).detach().cpu().item()
    if cfg.conformers_file is not None:
        path = to_absolute_path(cfg.conformers_file)
        #loader = xyz_to_loader(path, cfg.eval_smiles, energy_model, batch_size=cfg.eval_batch_size)
        loader = xyz_to_loader(path, cfg.eval_smiles, energy_model)
        conformer_outputs = energy_model(
            next(iter(loader)).to(device), regularize=False
        )

        # print(conformer_outputs['energy'])
        # conformer_energies = output["energy"].detach().cpu().numpy()

    else:
        conformer_outputs = None
    Im = get_dataset_fig(
        states[0], outputs[0]["energy"], cfg, outputs=conformer_outputs
    )
    if cfg.visualize_conformations:
        visualize_conformations(
            states[0], outputs[0], atomic_number_table, n_samples=16
        )
    if cfg.conformers_file is not None:
        conformers_path = to_absolute_path(cfg.conformers_file)
        with open("soc.txt", "a+") as file:
            file.write(f"{soc_loss}\n")
    else:
        raise FileNotFoundError("we must have a cfg.conformers_file")

    if cfg.compute_coverage:
        cr, cp, threshold_range = eval_batch(
            states, energy_model, device, conformers_path, cfg.eval_smiles
        )
        cr = cr.tolist()
        with open("recall.csv", "a+", newline="") as file:
            write = csv.writer(file, delimiter=";")
            write.writerow(cr)
    return {"soc_loss": soc_loss, "energy_vis": Im}
