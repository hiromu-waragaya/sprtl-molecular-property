# -*- coding: utf-8 -*-
"""SMILES / RDKit Mol -> PyG Data（Sinple_GCN/single_gcn_train.py と同一ロジック）。"""
from __future__ import annotations

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def get_intervals(l):
    intervals = len(l) * [0]
    intervals[0] = 1
    for k in range(1, len(l)):
        intervals[k] = (len(l[k]) + 1) * intervals[k - 1]
    return intervals


def safe_index(l, e):
    try:
        return l.index(e)
    except Exception:
        return len(l)


class GraphConvConstants(object):
    possible_atom_list = [
        "C", "N", "O", "S", "F", "P", "Cl", "Mg", "Na", "Br", "Fe", "Ca",
        "Cu", "Mc", "Pd", "Pb", "K", "I", "Al", "Ni", "Mn",
    ]
    possible_numH_list = [0, 1, 2, 3, 4]
    possible_valence_list = [0, 1, 2, 3, 4, 5, 6]
    possible_formal_charge_list = [-3, -2, -1, 0, 1, 2, 3]
    possible_hybridization_list = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]
    possible_number_radical_e_list = [0, 1, 2]
    possible_chirality_list = ["R", "S"]
    reference_lists = [
        possible_atom_list,
        possible_numH_list,
        possible_valence_list,
        possible_formal_charge_list,
        possible_number_radical_e_list,
        possible_hybridization_list,
        possible_chirality_list,
    ]
    intervals = get_intervals(reference_lists)
    possible_bond_stereo = ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"]
    bond_fdim_base = 6


def atom_features(atom, bool_id_feat=False, explicit_H=False, use_chirality=False):
    if bool_id_feat:
        raise NotImplementedError("bool_id_feat=True is not supported")
    results_ = one_of_k_encoding_unk(
        atom.GetSymbol(),
        [
            "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca",
            "Fe", "As", "Al", "I", "B", "V", "K", "Tl", "Yb", "Sb", "Sn",
            "Ag", "Pd", "Co", "Se", "Ti", "Zn", "H", "Li", "Ge", "Cu", "Au",
            "Ni", "Cd", "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb", "Unknown",
        ],
    )
    results = (
        results_
        + one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        + one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6])
        + [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]
        + one_of_k_encoding_unk(
            atom.GetHybridization().name,
            [
                Chem.rdchem.HybridizationType.SP.name,
                Chem.rdchem.HybridizationType.SP2.name,
                Chem.rdchem.HybridizationType.SP3.name,
                Chem.rdchem.HybridizationType.SP3D.name,
                Chem.rdchem.HybridizationType.SP3D2.name,
            ],
        )
        + [atom.GetIsAromatic()]
    )
    if not explicit_H:
        results = results + one_of_k_encoding_unk(
            atom.GetTotalNumHs(), [0, 1, 2, 3, 4]
        )
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp("_CIPCode"), ["R", "S"]
            ) + [atom.HasProp("_ChiralityPossible")]
        except Exception:
            results = results + [False, False] + [atom.HasProp("_ChiralityPossible")]
    return np.array(results)


def bond_features(bond, use_chirality=False):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"],
        )
    return np.array(bond_feats)


def get_bond_pair(mol):
    bonds = mol.GetBonds()
    res = [[], []]
    for bond in bonds:
        res[0] += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        res[1] += [bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]
    return res


def mol2vec(mol):
    """参考 single_gcn_train.py と同一のグラフ構築（結合なし分子も扱う）。"""
    atoms = mol.GetAtoms()
    bonds = mol.GetBonds()
    node_f = [atom_features(atom) for atom in atoms]
    edge_index = get_bond_pair(mol)
    edge_attr = [bond_features(bond, use_chirality=False) for bond in bonds]

    if len(edge_attr) == 0:
        return Data(
            x=torch.tensor(node_f, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        )

    for bond in bonds:
        edge_attr.append(bond_features(bond))
    return Data(
        x=torch.tensor(node_f, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
        edge_attr=torch.tensor(edge_attr, dtype=torch.float),
    )
