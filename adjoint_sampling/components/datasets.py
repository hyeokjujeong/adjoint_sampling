# Copyright (c) Meta Platforms, Inc. and affiliates.

import os

import pickle
from functools import partial
from pathlib import Path
from typing import Optional, Union

import numpy as np

import torch
from ase import Atoms
from ase.io import read
from ase.optimize.lbfgs import LBFGS

from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from rdkit.Chem.AllChem import EmbedMultipleConfs
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from tqdm import tqdm

from adjoint_sampling.components.ourjoblib import joblib_map
from adjoint_sampling.energies.fairchem_energy import FairChemEnergy
from adjoint_sampling.sampletorsion.featurize import drugs_types, featurize_mol
from adjoint_sampling.sampletorsion.rotate import get_index_to_rotate
from adjoint_sampling.sampletorsion.torsion import get_tor_indexes_daniel, T_TOR_INDEXES

from adjoint_sampling.utils.data_utils import get_atomic_graph


# Define covalent radii (in Å) for relevant elements
COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "O": 0.66,
    "N": 0.71,
    "F": 0.57,
    "Cl": 0.99,
    "Br": 1.14,
    "S": 1.05,
    "I": 1.33,
    "P": 1.07,
}

VAN_DER_WAALS_RADII = {
    "H": 1.10,
    "C": 1.70,
    "O": 1.52,
    "N": 1.55,
    "F": 1.47,
    "Cl": 1.75,
    "Br": 1.83,
    "S": 1.80,
    "I": 1.98,
    "P": 1.80,
}

# Atom-specific tolerances (in Å) for each element
ATOM_SPECIFIC_TOLERANCES = {
    "H": 0.1,
    "C": 0.15,
    "O": 0.12,
    "N": 0.12,
    "F": 0.1,
    "Cl": 0.2,
    "Br": 0.2,
    "S": 0.18,
    "I": 0.2,
    "P": 0.18,
}

# Empirical multipliers for bond types
BOND_MULTIPLIERS = {
    Chem.rdchem.BondType.SINGLE: 1.0,
    Chem.rdchem.BondType.DOUBLE: 0.85,
    Chem.rdchem.BondType.TRIPLE: 0.75,
    Chem.rdchem.BondType.AROMATIC: 0.95,
}


def get_covalent_radius(atom_symbol):
    """Returns the covalent radius for a given atom symbol."""
    return COVALENT_RADII.get(atom_symbol, None)


def get_van_der_waals_radius(atom_symbol):
    """Returns the covalent radius for a given atom symbol."""
    return VAN_DER_WAALS_RADII.get(atom_symbol, None)


def get_atom_specific_tolerance(atom_symbol):
    """Returns the tolerance specific to a given atom symbol."""
    return ATOM_SPECIFIC_TOLERANCES.get(atom_symbol, 0.1)


def bond_length_matrix(rdmol):
    if rdmol is None:
        raise ValueError("Invalid SMILES string.")

    # Get number of atoms (including hydrogens)
    N = rdmol.GetNumAtoms()

    # Initialize the NxN matrix with 'inf' for non-bonded pairs
    length_matrix = torch.full((N, N), fill_value=np.nan)
    length_matrix.fill_diagonal_(0.0)
    type_matrix = torch.full((N, N), fill_value=0)
    # type_matrix.fill_diagonal_(0.0)
    # Create a list to store atom symbols in order
    atomic_numbers = [rdmol.GetAtomWithIdx(i).GetAtomicNum() for i in range(N)]

    # Iterate over bonds in the molecule

    for bond in rdmol.GetBonds():
        # Get the indices of the bonded atoms
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        # Get the atom symbols for each atom
        atom1, atom2 = (
            rdmol.GetAtomWithIdx(i).GetSymbol(),
            rdmol.GetAtomWithIdx(j).GetSymbol(),
        )

        # Get bond multiplier
        multiplier = 1.5  # * BOND_MULTIPLIERS.get(bond.GetBondType(), 1.0)

        # Get covalent radii and atom-specific tolerances
        # Get radii and tolerances
        radius1, radius2 = get_covalent_radius(atom1), get_covalent_radius(atom2)
        # Calculate the upper limit of bond length based on bond type
        if radius1 is not None and radius2 is not None:
            # check with the chemistry team
            bond_length = (radius1 + radius2) * multiplier
            length_matrix[i, j] = bond_length
            length_matrix[j, i] = bond_length  # Matrix is symmetric

            type_matrix[i, j] = 1
            type_matrix[j, i] = 1

    # fill out non-bond edges with limits
    for i in range(length_matrix.shape[0]):
        for j in range(i, length_matrix.shape[0]):

            if torch.isnan(length_matrix[i, j]):
                atom1, atom2 = (
                    rdmol.GetAtomWithIdx(i).GetSymbol(),
                    rdmol.GetAtomWithIdx(j).GetSymbol(),
                )
                multiplier = (
                    1.0 / 1.5
                )  # (1.2  * BOND_MULTIPLIERS.get(bond.GetBondType(), 1.0) )

                # Get covalent radii and atom-specific tolerances
                # Get radii and tolerances
                radius1, radius2 = get_van_der_waals_radius(
                    atom1
                ), get_van_der_waals_radius(atom2)

                # Calculate the upper limit of bond length based on bond type
                if radius1 is not None and radius2 is not None:
                    # check with the chemistry team
                    bond_length = (radius1 + radius2) * multiplier
                    length_matrix[i, j] = bond_length
                    length_matrix[j, i] = bond_length  # Matrix is symmetric
    # print(matrix)
    return length_matrix, type_matrix, atomic_numbers


def get_positions(
    mol: Chem.Mol, tor_indexes: T_TOR_INDEXES, relax: bool, energy_model=None
) -> np.ndarray:
    """provides the positions of a conformation with dihedrals set to zero"""
    # find a conformation
    EmbedMultipleConfs(
        mol,
        numConfs=1,
        randomSeed=42,  # for reproducibility
        pruneRmsThresh=-1,  # Remove similar conformers
        enforceChirality=True,
    )
    confId = 0
    conf = mol.GetConformer(confId)

    # relax structure
    if relax:
        atoms = Atoms(
            numbers=[atom.GetAtomicNum() for atom in mol.GetAtoms()],
            positions=conf.GetPositions(),
        )
        if energy_model is None:
            raise ValueError(
                "you must provide an energy model with an ASE calculator in energy_model.calc"
            )
        atoms.calc = energy_model.calc
        out = LBFGS(atoms)
        out.run()
        conf.SetPositions(out.atoms.get_positions())

    # set dihedrals to zero
    for tor_ind in tor_indexes:
        rdMolTransforms.SetDihedralRad(conf, *tor_ind, 0.0)
    return conf.GetPositions()


def tordiff_featurizer(
    smile: Union[str, bytes],
    mol: Chem.Mol,
) -> Data:
    # our additions
    if isinstance(smile, bytes):
        smile = smile.decode("utf-8")

    # mostly copied
    if "." in smile:
        raise ValueError("tordiff cant have dots in smiles")

    N = mol.GetNumAtoms()

    if N < 4:
        raise ValueError("mol_too_small")

    canonical_smi = Chem.MolToSmiles(mol, isomericSmiles=False)

    data = featurize_mol(mol, drugs_types)
    if not data:
        raise ValueError("featurize_mol_failed")

    data.canonical_smi = canonical_smi
    # data.mol = mol
    return data


def compute_torsional_data(
    rdmol: Chem.Mol,
    smiles: str,
    energy_model: torch.nn.Module,
    relax: bool,
) -> tuple[np.ndarray, dict, Data]:
    # torsions
    Chem.GetSymmSSSR(rdmol)  # helps with spice
    data_tordiff = tordiff_featurizer(smiles, rdmol)
    rdkit_edge_index = data_tordiff["edge_index"].clone()
    tor_index = get_tor_indexes_daniel(rdmol)

    n_torsions = len(tor_index)
    if n_torsions == 0:
        raise ValueError("no torsions")

    # need to run this before we convert to LongTensor
    positions = get_positions(
        rdmol, tor_indexes=tor_index, relax=relax, energy_model=energy_model
    )

    tor_index = torch.LongTensor(tor_index).T
    torsions = torch.zeros(n_torsions)

    # remove / adjust data we can't use
    del data_tordiff.pos  # we don't want two positions sitting around
    del data_tordiff.edge_index  # we copied this to rdkit_edge_index
    #
    # estimator defines `bonds = data.edge_index[:, data.edge_mask].long()`
    # estimator gets `bond_vec = data.positions[bonds[1]] - data.positions[bonds[0]]`
    # tordiff rotates `rot_vec = data.positions[bonds[0]] - data.positions[bonds[1]]` "NOTE: different from paper"
    # we rotate around `rot_axes = positions[tor_index[2]] - positions[tor_index[1]]` with origin `positions[tor_index[1]]`
    #
    # our mask is true for edges that = rot_axis, and false otherwise!
    # this will be in a different order than tor_index, in general
    edge_mask = torch.stack(
        [(ti == rdkit_edge_index.T).all(dim=-1) for ti in tor_index[1:3, :].T]
    ).any(dim=0)

    index_to_rotate = get_index_to_rotate(tor_index, rdkit_edge_index)

    return (
        positions,
        dict(
            learn_torsions=True,
            tor_index=tor_index,
            index_to_rotate=index_to_rotate,
            torsions=torsions,
            n_torsions=n_torsions,
            tor_per_mol_label=torch.arange(n_torsions),
            rdkit_edge_index=rdkit_edge_index,
            edge_mask=edge_mask,
        ),
        data_tordiff,
    )


# generates dataloader of all zeros for a single molecule type. Only need to generate one batch.
def get_homogeneous_dataset(
    smiles: str,
    energy_model,
    duplicate: int,
    learn_torsions: bool = False,
    relax: bool = False,
) -> list[Data]:
    """
    Args:
        learn_torsions: If True, positions are angles and x_positions are a conformer with dihedral angles zero, else zeros
        relax: if torsions is true, relax the conformer before rotating dihedrals to zero
    """
    # import ipdb; ipdb.set_trace()
    atomic_number_table = energy_model.atomic_numbers
    atomic_index_table = {int(z): i for i, z in enumerate(atomic_number_table)}

    rdmol = Chem.MolFromSmiles(smiles)
    rdmol = Chem.AddHs(rdmol)

    length_matrix, type_matrix, atom_list = bond_length_matrix(rdmol)

    if learn_torsions:
        positions, tor_data, data_tordiff = compute_torsional_data(
            rdmol=rdmol,
            smiles=smiles,
            energy_model=energy_model,
            relax=relax,
        )
    else:
        positions = torch.zeros(len(atom_list), 3)
        tor_data = {}
        data_tordiff = Data()

    dataset = []
    for _ in range(duplicate):
        data_i = get_atomic_graph(atom_list, positions, atomic_index_table)
        edge_index = data_i["edge_index"]
        length_attr = length_matrix[edge_index[0], edge_index[1]].unsqueeze(-1)
        type_attr = type_matrix[edge_index[0], edge_index[1]].unsqueeze(-1)
        edge_attr = torch.cat([length_attr, type_attr], dim=-1).float()
        data_i["edge_attrs"] = edge_attr
        data_i["smiles"] = smiles

        data_tordiff_dup = data_tordiff.clone()

        # no overlap between these keys
        assert set(data_i.keys()).intersection(tor_data.keys()) == set()
        for k, v in tor_data.items():
            data_i[k] = v

        # no overlap between these keys either
        assert set(data_tordiff_dup.keys()).intersection(set(data_i.keys())) == set()
        for k in data_tordiff_dup.keys():
            if isinstance(data_tordiff_dup[k], np.ndarray):
                data_i[k] = torch.from_numpy(data_tordiff_dup[k])
            else:
                data_i[k] = data_tordiff_dup[k]

        dataset.append(data_i)

    return dataset


def process_smiles(
    smiles: str,
    learn_torsions: bool,
    energy_model,
    relax: bool,
    duplicate: int,
    atomic_index_table,
    r_max,
) -> Optional[Data]:
    try:
        rdmol = Chem.MolFromSmiles(smiles)
        rdmol = Chem.AddHs(rdmol)
        length_matrix, type_matrix, atom_list = bond_length_matrix(rdmol)
    except:  # noqa: E722
        print("smiles invalid")
        return None

    if learn_torsions:
        try:
            positions, tor_data, data_tordiff = compute_torsional_data(
                rdmol=rdmol,
                smiles=smiles,
                energy_model=energy_model,
                relax=relax,
            )
        except ValueError as e:
            print(e)
            return None
    else:
        positions = torch.zeros((len(atom_list), 3))
        tor_data = {}
        data_tordiff = Data()

    for _ in range(duplicate):

        try:
            data_i = get_atomic_graph(atom_list, positions, atomic_index_table)

        except ValueError:
            print("atom not available")
            return None

        edge_index = data_i["edge_index"]
        length_attr = length_matrix[edge_index[0], edge_index[1]].unsqueeze(-1)
        type_attr = type_matrix[edge_index[0], edge_index[1]].unsqueeze(-1)
        edge_attr = torch.cat([length_attr, type_attr], dim=-1).float()
        data_i["edge_attrs"] = edge_attr
        data_i["smiles"] = smiles

        data_tordiff_dup = data_tordiff.clone()

        # no overlap between these keys
        assert set(data_i.keys()).intersection(tor_data.keys()) == set()
        for k, v in tor_data.items():
            data_i[k] = v

        # no overlap between these keys either
        assert set(data_tordiff_dup.keys()).intersection(set(data_i.keys())) == set()
        for k in data_tordiff_dup.keys():
            if isinstance(data_tordiff_dup[k], np.ndarray):
                data_i[k] = torch.from_numpy(data_tordiff_dup[k])
            else:
                data_i[k] = data_tordiff_dup[k]

        return data_i


def get_spice_dataset(
    dataset_path: str,
    energy_model,
    duplicate: int,
    torsion_cache_suffix: str,
    learn_torsions: bool,
    relax: bool,
    cache_only: bool = False,
):

    atomic_number_table = energy_model.atomic_numbers
    atomic_index_table = {int(z): i for i, z in enumerate(atomic_number_table)}

    r_max = energy_model.r_max

    if not os.path.isabs(dataset_path):
        repo_dir = Path(__file__).resolve().parent.parent.parent
        dataset_path = os.path.join(repo_dir, dataset_path)

    if isinstance(energy_model, FairChemEnergy):
        energy_model_suffix = "_fc"
    else:
        ValueError("energy_model is unrecognized")
    if learn_torsions:
        cache_path = (
            os.path.splitext(dataset_path)[0]
            + energy_model_suffix
            + torsion_cache_suffix
            + ".pkl"
        )
    else:
        cache_path = os.path.splitext(dataset_path)[0] + energy_model_suffix + ".pkl"
    if Path(cache_path).is_file():
        print("found cached dataset!")
        if cache_only:
            return
        else:
            print(f"loading from {cache_path}")
            with open(cache_path, "rb") as handle:
                dataset = pickle.load(handle)
            return dataset
    with open(dataset_path, "r") as f:
        next(f)
        train_mols = f.readlines()

    _, smiles_list_in, _ = zip(
        *[line.strip().split() for line in tqdm(train_mols, desc="mol text")]
    )

    print(f"loading from {dataset_path}")

    todo = partial(
        process_smiles,
        learn_torsions=learn_torsions,
        energy_model=energy_model,
        relax=relax,
        duplicate=duplicate,
        atomic_index_table=atomic_index_table,
        r_max=r_max,
    )
    dataset = joblib_map(
        todo,
        smiles_list_in,
        n_jobs=int(os.environ["SLURM_CPUS_ON_NODE"])
        if os.environ.get("SLURM_CPUS_ON_NODE", None)
        else 1,
        inner_max_num_threads=1,
        desc="reading smiles",
        total=len(smiles_list_in),
    )
    dataset = [item for item in dataset if item is not None]

    print("caching data to ", cache_path)
    with open(cache_path, "wb") as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dataset


def test_dataset(
    smiles_list,
    energy_model,
    duplicate: int = 1,
    learn_torsions: bool = False,
):
    atomic_number_table = energy_model.atomic_numbers
    atomic_index_table = {int(z): i for i, z in enumerate(atomic_number_table)}

    dataset = []
    for smiles in smiles_list:
        rdmol = Chem.MolFromSmiles(smiles)
        rdmol = Chem.AddHs(rdmol)
        length_matrix, type_matrix, atom_list = bond_length_matrix(rdmol)

        if learn_torsions:
            try:
                positions, tor_data, data_tordiff = compute_torsional_data(
                    rdmol=rdmol,
                    smiles=smiles,
                    energy_model=energy_model,
                    relax=False,
                )
            except ValueError as e:
                print(e)
                continue
        else:
            positions = torch.zeros((len(atom_list), 3))
            tor_data = {}
            data_tordiff = Data()

        for _ in range(duplicate):
            data_i = get_atomic_graph(atom_list, positions, atomic_index_table)
            edge_index = data_i["edge_index"]
            length_attr = length_matrix[edge_index[0], edge_index[1]].unsqueeze(-1)
            type_attr = type_matrix[edge_index[0], edge_index[1]].unsqueeze(-1)
            edge_attr = torch.cat([length_attr, type_attr], dim=-1).float()
            data_i["edge_attrs"] = edge_attr
            data_i["smiles"] = smiles

            data_tordiff_dup = data_tordiff.clone()

            # no overlap between these keys
            assert set(data_i.keys()).intersection(tor_data.keys()) == set()
            for k, v in tor_data.items():
                data_i[k] = v

            # no overlap between these keys either
            assert (
                set(data_tordiff_dup.keys()).intersection(set(data_i.keys())) == set()
            )
            for k in data_tordiff_dup.keys():
                if isinstance(data_tordiff_dup[k], np.ndarray):
                    data_i[k] = torch.from_numpy(data_tordiff_dup[k])
                else:
                    data_i[k] = data_tordiff_dup[k]

            dataset.append(data_i)

    return dataset


def xyz_to_loader(filename, smiles, energy_model, batch_size = None):
    atomic_number_table = energy_model.atomic_numbers
    atomic_index_table = {int(z): i for i, z in enumerate(atomic_number_table)}

    molecules = read(filename, index=":")
    rdmol = Chem.MolFromSmiles(smiles)
    rdmol = Chem.AddHs(rdmol)
    atomic_numbers = [
        rdmol.GetAtomWithIdx(i).GetAtomicNum() for i in range(rdmol.GetNumAtoms())
    ]
    data_set = []

    for mol in molecules:
        positions = torch.tensor(mol.get_positions(), dtype=torch.get_default_dtype())
        data_i = get_atomic_graph(atomic_numbers, positions, atomic_index_table)
        data_set.append(data_i)

    data_loader = DataLoader(
        dataset=data_set,
        batch_size=min(batch_size, len(data_set)) if batch_size is not None else len(data_set),
        shuffle=False,
        drop_last=False,
    )
    return data_loader
