# Copyright (c) Meta Platforms, Inc. and affiliates.

import csv
import io
import os
from itertools import product
from pathlib import Path
from typing import Union

import networkx as nx
import numpy as np
import torch
from ase import Atoms
from ase.optimize.lbfgs import LBFGS
from openbabel import openbabel as ob
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdDetermineBonds
from tqdm import tqdm

from adjoint_sampling.components.ourjoblib import joblib_map
from adjoint_sampling.components.sde import (
    euler_maruyama_step,
    quadratic_linear_discretization,
    uniform_discretization,
)


# Constants
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


def print_atoms(molecule):
    for i, atom in enumerate(molecule.GetAtoms()):
        positions = molecule.GetConformer().GetAtomPosition(i)
        print(atom.GetSymbol(), positions.x, positions.y, positions.z)


def setup_device(args):
    """Setup compute device."""
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(
            f"{args.device}:{args.gpu_id}" if args.device == "cuda" else args.device
        )
    print(f"Using device: {device}")
    return device


def get_mol_from_mol_block(mol_block: str) -> Chem.Mol:
    rdkit_mol = Chem.MolFromMolBlock(mol_block, removeHs=False)
    if rdkit_mol is None:
        rdkit_mol = Chem.MolFromMolBlock(mol_block, removeHs=False, sanitize=False)
        if rdkit_mol is not None:
            try:
                Chem.SanitizeMol(
                    rdkit_mol, sanitizeOps=Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES
                )
            except Exception as e:
                print(f"Sanitization failed: {str(e)}")
    return rdkit_mol


def get_networkx_graph(mol: Chem.Mol) -> tuple[nx.Graph, dict[int, int]]:
    m = {atom.GetIdx(): atom.GetAtomicNum() for atom in mol.GetAtoms()}
    g = nx.convert.from_edgelist(
        [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()]
    ).to_undirected()
    g = nx.relabel_nodes(g, m)
    return g, m


def get_map_if_atoms_same_and_undirected_isomorphism(
    mol1: Chem.Mol, mol2: Chem.Mol
) -> list[list[tuple[int, int]]] | None:
    g1, m1 = get_networkx_graph(mol1)
    g2, m2 = get_networkx_graph(mol2)

    if nx.is_isomorphic(g1, g2, node_match=lambda x, y: x == y):
        assert len(m1) == len(m2)
        maps = [[(i, j) for i, j in zip(m1, m2)]]
        return maps  # using rdkit's maps signature. https://github.com/rdkit/rdkit-orig/blob/57058c886a49cc597b0c40641a28697ee3a57aee/rdkit/Chem/AllChem.py#L160
    else:
        return None


def read_rdkit_mols(
    smiles,
    batch_size: int,
    optimize: bool = False,
    relax: bool = False,
    calc: bool = None,
    fmax: float = 0.05,
) -> list[Chem.Mol]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Failed to convert SMILES to RDKit molecule: {smiles}")
        return None
    mol = Chem.AddHs(mol)
    # Generate conformers
    AllChem.EmbedMultipleConfs(
        mol,
        numConfs=batch_size,
        randomSeed=42,  # for reproducibility
        pruneRmsThresh=-1,  # Remove similar conformers
        enforceChirality=True,
    )

    if optimize:
        for conf_id in range(mol.GetNumConformers()):
            AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)

    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats("xyz", "mol")

    mol_list = []
    for conf in mol.GetConformers():
        new_mol = Chem.Mol(mol)
        new_mol.RemoveAllConformers()
        new_mol.AddConformer(conf)
        atom_list = [atom.GetAtomicNum() for atom in new_mol.GetAtoms()]
        positions = conf.GetPositions()
        atoms = Atoms(numbers=atom_list, positions=np.array(positions))
        if relax:
            atoms.calc = calc
            opt = LBFGS(atoms)
            opt.run(fmax=fmax, steps=1000)

            num_atoms = len(atoms.positions)
            xyz_block = f"{num_atoms}\nGenerated conformer\n"
            atomic_number = atoms.numbers
            positions = atoms.positions
            for j in range(num_atoms):
                atom_symbol = SYM_LIST[atomic_number[j]]
                xyz_block += f"{atom_symbol} {positions[j, 0]:.6f} {positions[j, 1]:.6f} {positions[j, 2]:.6f}\n"
            obmol = ob.OBMol()
            obConversion.ReadString(obmol, xyz_block)
            # Perceive bonds
            obmol.ConnectTheDots()  # Connect atoms based on distance
            obmol.PerceiveBondOrders()  # Perceive bond orders
            obmol.AssignSpinMultiplicity(True)  # Clean up radical centers
            mol_block = obConversion.WriteString(obmol)
            rdkit_mol = get_mol_from_mol_block(mol_block)
            mol_list.append(rdkit_mol)
        else:
            mol_list.append(new_mol)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return mol_list


def read_rdkit_mols_with_rdkit(
    smiles,
    batch_size,
    optimize=False,
    relax=False,
    calc=None,
    fmax=0.05,
    charge=0,
):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Failed to convert SMILES to RDKit molecule: {smiles}")
        return None
    mol = Chem.AddHs(mol)
    # Generate conformers
    AllChem.EmbedMultipleConfs(
        mol,
        numConfs=batch_size,
        randomSeed=42,  # for reproducibility
        pruneRmsThresh=-1,  # Remove similar conformers
        enforceChirality=True,
    )

    if optimize:
        for conf_id in range(mol.GetNumConformers()):
            AllChem.MMFFOptimizeMolecule(mol, confId=conf_id)

    mol_list = []
    for conf in mol.GetConformers():
        new_mol = Chem.Mol(mol)
        new_mol.RemoveAllConformers()
        new_mol.AddConformer(conf)
        atom_list = [atom.GetAtomicNum() for atom in new_mol.GetAtoms()]
        positions = conf.GetPositions()
        atoms = Atoms(numbers=atom_list, positions=np.array(positions))
        if relax:
            atoms.calc = calc
            opt = LBFGS(atoms)
            opt.run(fmax=fmax, steps=1000)

            num_atoms = len(atoms.positions)
            xyz_block = f"{num_atoms}\nGenerated conformer\n"
            atomic_number = atoms.numbers
            positions = atoms.positions
            for j in range(num_atoms):
                atom_symbol = SYM_LIST[atomic_number[j]]
                xyz_block += f"{atom_symbol} {positions[j, 0]:.6f} {positions[j, 1]:.6f} {positions[j, 2]:.6f}\n"
            rdkit_mol = Chem.MolFromXYZBlock(xyz_block)
            rdDetermineBonds.DetermineBonds(rdkit_mol, charge=charge)
            mol_list.append(rdkit_mol)
        else:
            mol_list.append(new_mol)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return mol_list


def read_xyz_files(xyz_path: Union[Path, str]) -> list[Chem.Mol]:
    """Read XYZ file using OpenBabel and convert to RDKit molecules"""
    ref_mols = []

    # Create OpenBabel objects
    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats("xyz", "mol")

    with open(xyz_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):  # and len(ref_mols) < max_n_refs:
        line = lines[i].strip()
        # Check if line starts with a number (number of atoms)
        if line and line[0].isdigit():
            num_atoms = int(line)
            energy = float(lines[i + 1])
            # energy_list.append(energy)
            positions = []
            atom_list = []

            # Create temporary XYZ file for this conformer
            xyz_block = f"{num_atoms}\n{energy}\n"
            for line in lines[i + 2 : i + 2 + num_atoms]:
                xyz_block += line
                position = np.array(line.split()[1:4], dtype=float)
                positions.append(position)
                atom_symbol = line.split()[0]
                atom_number = ATOMIC_NUMBERS[atom_symbol]
                atom_list.append(atom_number)

            # Create OpenBabel molecule
            obmol = ob.OBMol()
            obConversion.ReadString(obmol, xyz_block)

            # Perceive bonds
            obmol.ConnectTheDots()  # Connect atoms based on distance
            obmol.PerceiveBondOrders()  # Perceive bond orders

            # Convert to RDKit molecule
            mol_block = obConversion.WriteString(obmol)
            rdkit_mol = get_mol_from_mol_block(mol_block)

            if rdkit_mol is not None:
                ref_mols.append(rdkit_mol)
                del rdkit_mol
            else:
                print(f"Failed to convert conformer at index {i}")

            i += num_atoms + 2
        else:
            i += 1

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return ref_mols


def read_xyz_files_with_rdkit(xyz_path, charge):
    ref_mols = []

    with open(xyz_path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line and line[0].isdigit():
            num_atoms = int(line)
            energy = float(lines[i + 1])
            xyz_block = f"{num_atoms}\n{energy}\n"
            for line in lines[i + 2 : i + 2 + num_atoms]:
                xyz_block += line

            rdkit_mol = Chem.MolFromXYZBlock(xyz_block)
            rdDetermineBonds.DetermineBonds(rdkit_mol, charge=charge)
            if rdkit_mol is not None:
                ref_mols.append(rdkit_mol)
            else:
                print(f"Failed to convert conformer at index {i} for {xyz_path}")
            i += num_atoms + 2
        else:
            i += 1

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return ref_mols


def generate_conformers(
    graph_state,
    atomic_numbers: torch.LongTensor,
    relax: bool = False,
    calc=None,
    fmax: float = 0.05,
    steps: int = 1_000,
) -> list[Chem.Mol]:
    """
    Generate conformers from graph state using OpenBabel.

    if relax=True,
    """
    atomic_numbers = atomic_numbers.detach().cpu()
    positions = graph_state["positions"].detach().cpu()
    atom_list = graph_state["node_attrs"].detach().cpu()

    relaxed_positions = []
    if relax:
        ptr = 0
        for i in range(len(graph_state["ptr"]) - 1):
            num_atoms = graph_state["ptr"][i + 1] - graph_state["ptr"][i]
            atom_list_i = atom_list[ptr : ptr + num_atoms]
            atomic_number_i = atomic_numbers[torch.nonzero(atom_list_i)[:, 1]]
            positions_i: torch.Tensor = positions[ptr : ptr + num_atoms]
            ptr += num_atoms
            atoms = Atoms(
                numbers=np.array(atomic_number_i),
                positions=np.array(
                    positions_i.float()
                ),  # note sending positions to float32!
            )
            atoms.calc = calc
            opt = LBFGS(atoms)
            opt.run(fmax=fmax, steps=steps)
            relaxed_positions_i = atoms.positions
            relaxed_positions.append(torch.from_numpy(relaxed_positions_i))
        relaxed_positions = torch.cat(relaxed_positions, dim=0)
        positions = relaxed_positions.float()

    gen_mols = []
    obConversion = ob.OBConversion()
    obConversion.SetInAndOutFormats("xyz", "mol")
    ij = 0
    for i in range(len(graph_state["ptr"]) - 1):
        num_atoms = graph_state["ptr"][i + 1] - graph_state["ptr"][i]

        # Create XYZ block for this molecule
        xyz_block = f"{num_atoms.item()}\nGenerated conformer\n"

        # Add atoms and their coordinates
        for j in range(num_atoms):
            atomic_number = atomic_numbers[torch.nonzero(atom_list[ij])]
            atom_symbol = SYM_LIST[atomic_number[0, 0].item()]
            xyz_block += f"{atom_symbol} {positions[ij, 0].item():.6f} {positions[ij, 1].item():.6f} {positions[ij, 2].item():.6f}\n"
            ij += 1
        try:
            # Convert XYZ to molecule using OpenBabel
            obmol = ob.OBMol()
            obConversion.ReadString(obmol, xyz_block)
            # Perceive bonds
            obmol.ConnectTheDots()  # Connect atoms based on distance
            obmol.PerceiveBondOrders()  # Perceive bond orders
            obmol.AssignSpinMultiplicity(True)  # Clean up radical centers

            # Convert to RDKit molecule
            mol_block = obConversion.WriteString(obmol)
            rdkit_mol = get_mol_from_mol_block(mol_block)

            if rdkit_mol is not None:
                gen_mols.append(rdkit_mol)
                del rdkit_mol, obmol, mol_block
            else:
                print(f"Failed to convert generated molecule {i}")
        except Exception as e:
            print(f"Error generating molecule {i}: {e}")
            continue
    print("generated {} mols".format(len(gen_mols)))
    return gen_mols


def generate_conformers_with_rdkit(
    graph_state,
    atomic_numbers,
    relax=False,
    calc=None,
    fmax=0.05,
    steps=1_000,
    charge=0,
):
    atomic_numbers = atomic_numbers.detach().cpu()
    positions = graph_state["positions"].detach().cpu()
    atom_list = graph_state["node_attrs"].detach().cpu()

    relaxed_positions = []
    if relax:
        ptr = 0
        for i in range(len(graph_state["ptr"]) - 1):
            num_atoms = graph_state["ptr"][i + 1] - graph_state["ptr"][i]
            atom_list_i = atom_list[ptr : ptr + num_atoms]
            atomic_number_i = atomic_numbers[torch.nonzero(atom_list_i)[:, 1]]
            positions_i = positions[ptr : ptr + num_atoms]
            ptr += num_atoms
            atoms = Atoms(
                numbers=np.array(atomic_number_i), positions=np.array(positions_i)
            )
            atoms.calc = calc
            opt = LBFGS(atoms)
            opt.run(fmax=fmax, steps=steps)
            relaxed_positions_i = atoms.positions
            relaxed_positions.append(torch.from_numpy(relaxed_positions_i))
        relaxed_positions = torch.cat(relaxed_positions, dim=0)
        positions = relaxed_positions.float()

    gen_mols = []
    ij = 0
    for i in range(len(graph_state["ptr"]) - 1):
        num_atoms = graph_state["ptr"][i + 1] - graph_state["ptr"][i]
        xyz_block = f"{num_atoms.item()}\nGenerated conformer\n"
        for j in range(num_atoms):
            atomic_number = atomic_numbers[torch.nonzero(atom_list[ij])]
            atom_symbol = SYM_LIST[atomic_number[0, 0].item()]
            xyz_block += f"{atom_symbol} {positions[ij, 0].item():.6f} {positions[ij, 1].item():.6f} {positions[ij, 2].item():.6f}\n"
            ij += 1
        try:
            rdkit_mol = Chem.MolFromXYZBlock(xyz_block)
            rdDetermineBonds.DetermineBonds(rdkit_mol, charge=charge)
            if rdkit_mol is not None:
                gen_mols.append(rdkit_mol)
                del rdkit_mol
            else:
                print(f"Failed to convert generated molecule {i}")
        except Exception as e:
            print(f"Error generating molecule {i}: {e}")
            continue
    print("generated {} mols".format(len(gen_mols)))
    return gen_mols


def xyz_to_mol(block):
    lines = io.StringIO(block).readlines()
    num_atoms = int(lines[0])
    mol = Chem.Mol()
    edit_mol = Chem.EditableMol(mol)
    conf_data = []
    for i, line in enumerate(lines[2 : num_atoms + 2]):
        parts = line.split()
        atom_symbol, x, y, z = (
            parts[0],
            float(parts[1]),
            float(parts[2]),
            float(parts[3]),
        )
        if atom_symbol != "H":
            conf_data.append((atom_symbol, x, y, z))

    num_atoms = len(conf_data)
    conf = Chem.Conformer(num_atoms)
    for i in range(num_atoms):
        atom_symbol, x, y, z = conf_data[i]
        atom = Chem.Atom(atom_symbol)
        atom_idx = edit_mol.AddAtom(atom)
        conf.SetAtomPosition(atom_idx, (x, y, z))

    mol = edit_mol.GetMol()
    mol.AddConformer(conf)
    Chem.SanitizeMol(mol)
    return mol


def safe_remove_hydrogens(mol: Chem.Mol) -> Chem.Mol:
    try:
        return Chem.RemoveAllHs(mol)
    except:  # noqa: E722
        return Chem.RemoveAllHs(mol, sanitize=False)


def calc_rmsd_pairwise(gen_mol: Chem.Mol, ref_mol: Chem.Mol) -> np.ndarray:
    gen_mol_noH = safe_remove_hydrogens(gen_mol)
    ref_mol_noH = safe_remove_hydrogens(ref_mol)
    # only if there are the same number of atoms
    if ref_mol_noH.GetNumAtoms() == gen_mol_noH.GetNumAtoms():
        try:
            # automatically find pairs
            rms_dist = AllChem.GetBestRMS(gen_mol_noH, ref_mol_noH)
            return rms_dist
        except RuntimeError:
            pass

        try:
            # use our pairs
            maps = get_map_if_atoms_same_and_undirected_isomorphism(
                gen_mol_noH, ref_mol_noH
            )
            rms_dist = AllChem.GetBestRMS(gen_mol_noH, ref_mol_noH, map=maps)
            return rms_dist
        except RuntimeError:
            pass
    return np.nan


def calc_rmsd_parallel(
    gen_mols: list[Chem.Mol], ref_mols: list[Chem.Mol]
) -> np.ndarray:
    gen_mols = list(map(safe_remove_hydrogens, gen_mols))
    ref_mols = list(map(safe_remove_hydrogens, ref_mols))
    work = product(ref_mols, gen_mols)
    rmsds = joblib_map(
        lambda x: calc_rmsd_pairwise(*x),
        work,
        n_jobs=int(os.environ["SLURM_CPUS_ON_NODE"])
        if os.environ.get("SLURM_CPUS_ON_NODE", False)
        else 1,
        inner_max_num_threads=1,
        desc="computing rmsd",
        total=len(ref_mols) * len(gen_mols),
    )
    return np.asarray(rmsds).reshape(len(ref_mols), len(gen_mols))


def calc_rmsd(gen_mols: list[Chem.Mol], ref_mols: list[Chem.Mol]) -> np.ndarray:
    rmsd_array = np.full((len(ref_mols), len(gen_mols)), np.inf)
    for i, ref_mol in enumerate(tqdm(ref_mols)):
        for j, gen_mol in enumerate(gen_mols):
            ref_mol_noH = safe_remove_hydrogens(ref_mol)
            gen_mol_noH = safe_remove_hydrogens(gen_mol)

            # only if there are the same number of atoms
            if ref_mol_noH.GetNumAtoms() == gen_mol_noH.GetNumAtoms():
                try:
                    try:
                        # automatically find pairs
                        rms_dist = AllChem.GetBestRMS(gen_mol_noH, ref_mol_noH)
                    except RuntimeError:
                        maps = get_map_if_atoms_same_and_undirected_isomorphism(
                            gen_mol_noH, ref_mol_noH
                        )
                        rms_dist = AllChem.GetBestRMS(
                            gen_mol_noH, ref_mol_noH, map=maps
                        )
                    rmsd_array[i, j] = rms_dist
                except Exception:
                    rmsd_array[i, j] = np.nan
            else:
                rmsd_array[i, j] = np.nan
    return rmsd_array


def calc_performance_stats(rmsd_array, threshold, rsmd_array_name: str = ""):
    # Check if array is empty or all NaN
    if rmsd_array.size == 0 or np.all(np.isnan(rmsd_array)):
        msg = "Warning: Empty or all-NaN RMSD array"
        if rsmd_array_name:
            msg += f", named {rsmd_array_name}"
        print(msg)
        return None

    # Replace inf values with NaN for min operation
    rmsd_array = np.where(np.isinf(rmsd_array), np.nan, rmsd_array)

    coverage_recall = np.nanmean(
        np.nanmin(rmsd_array, axis=1, keepdims=True) < threshold, axis=0
    )
    amr_recall = np.nanmean(np.nanmin(rmsd_array, axis=1))
    coverage_precision = np.nanmean(
        np.nanmin(rmsd_array, axis=0, keepdims=True) < np.expand_dims(threshold, 1),
        axis=1,
    )
    amr_precision = np.nanmean(np.nanmin(rmsd_array, axis=0))

    return coverage_recall, amr_recall, coverage_precision, amr_precision


def visualize_conformers(ref_mols, gen_mols, n_samples, save_path):
    n_ref = min(len(ref_mols), n_samples)
    n_gen = min(len(gen_mols), n_samples)
    mols_to_draw = []
    legends = []
    for i in range(n_ref):
        ref_mol = ref_mols[i]
        mols_to_draw.append(ref_mol)
        legends.append(f"Ref {i+1}")
    for i in range(n_gen):
        mol = gen_mols[i]
        mols_to_draw.append(mol)
        legends.append(f"Gen {i+1}")
    img = Draw.MolsToGridImage(
        mols_to_draw,
        molsPerRow=4,
        subImgSize=(300, 300),
        legends=legends,
        returnPNG=False,
    )
    img.save(save_path)


def replace_none_with_nan(lst):
    return [np.nan if x is None else x for x in lst]


def replace_none_with_zero(lst):
    return [0.0 if x is None else x for x in lst]


def save_metrics(final_results, csv_path, none_processing="zero"):
    if none_processing == "nan":
        replace_fn_list = replace_none_with_nan
        replace_fn_number = lambda x: np.nan if np.isnan(x) else x
    elif none_processing == "zero":
        replace_fn_list = replace_none_with_zero
        replace_fn_number = lambda x: 0.0 if np.isnan(x) else x
    else:
        raise NotImplementedError

    threshold_ranges = final_results.iloc[0].threshold_ranges

    all_recall = final_results["recall"].apply(replace_fn_list)
    all_precision = final_results["precision"].apply(replace_fn_list)
    all_recall_crr = final_results["recall_crr"].apply(replace_fn_list)
    all_precision_crr = final_results["precision_crr"].apply(replace_fn_list)

    all_mr = final_results["mr"].apply(replace_fn_number)
    all_mp = final_results["mp"].apply(replace_fn_number)
    all_mr_crr = final_results["mr_crr"].apply(replace_fn_number)
    all_mp_crr = final_results["mp_crr"].apply(replace_fn_number)

    all_recall = np.nanmean(np.stack(all_recall), axis=0)
    all_precision = np.nanmean(np.stack(all_precision), axis=0)
    amr_recall = np.nanmean(all_mr)
    amr_precision = np.nanmean(all_mp)

    all_recall_c = np.nanmean(np.stack(all_recall_crr), axis=0)
    all_precision_c = np.nanmean(np.stack(all_precision_crr), axis=0)
    amr_recall_c = np.nanmean(all_mr_crr)
    amr_precision_c = np.nanmean(all_mp_crr)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "Threshold",
                "Coverage Recall",
                "Coverage Precision",
                "AMR Recall",
                "AMR Precision",
                "Coverage Recall SMILES",
                "Coverage Precision SMILES",
                "AMR Recall SMILES",
                "AMR Precision SMILES",
            ]
        )
        for i, thresh in enumerate(threshold_ranges):
            recall = all_recall[i]
            precision = all_precision[i]
            recall_c = all_recall_c[i]
            precision_c = all_precision_c[i]
            writer.writerow(
                [
                    thresh,
                    f"{recall * 100:.2f}",
                    f"{precision * 100:.2f}",
                    f"{amr_recall:.4f}",
                    f"{amr_precision:.4f}",
                    f"{recall_c * 100:.2f}",
                    f"{precision_c * 100:.2f}",
                    f"{amr_recall_c:.4f}",
                    f"{amr_precision_c:.4f}",
                ]
            )
    print(f"Metrics saved to {csv_path}")


@torch.no_grad()
def eval_batch(
    batch_list,
    energy_model,
    device,
    conformers_path,
    smiles,
):
    gen_mols = []
    for batch in batch_list:
        gen_mols.extend(
            generate_conformers(batch, energy_model.atomic_numbers.to(device))
        )

    xyz_path = conformers_path

    ref_mols = read_xyz_files(xyz_path)

    threshold_ranges = np.arange(0, 2.5, 0.125)
    rmsd_array = calc_rmsd(gen_mols, ref_mols)
    stats_ = calc_performance_stats(rmsd_array, threshold_ranges)

    if stats_ is not None:
        cr, _, cp, _ = stats_
    else:
        cr, cp = np.zeros(len(threshold_ranges)), np.zeros(len(threshold_ranges))
    return cr, cp, threshold_ranges


@torch.no_grad()
def integrate_sde_all_states(
    sde,
    graph0,
    num_integration_steps,
    discretization_scheme="uniform",
):

    if discretization_scheme == "uniform":
        times = uniform_discretization(num_steps=num_integration_steps)
    elif discretization_scheme == "ql":
        times = quadratic_linear_discretization(num_steps=num_integration_steps)
    else:
        raise ValueError(
            f"Unknown discretization_scheme option {discretization_scheme}"
        )

    graph_state = graph0.clone()
    graph_states = [graph_state.clone()]  # clone here
    controls = []
    for t, t_next in zip(times[:-1], times[1:]):
        dt = t_next - t
        u, graph_state = euler_maruyama_step(sde, t, graph_state, dt)
        controls.append(u)
        graph_states.append(graph_state.clone())
    return graph_states, controls
