import os
import json
import Bio.PDB
import Bio.PDB.Structure
import Bio.PDB.Chain
import Bio.PDB.Residue
import Bio.PDB.Atom
import Bio.SeqIO
from Bio.SeqIO.FastaIO import SimpleFastaParser
from typing import *

root = os.path.dirname(__file__)
project_root = os.path.dirname(root)

with open(os.path.join(project_root, 'utils/data/aa.json')) as f:
    aa_dict = json.load(f)


def get_project_root() -> str:
    return project_root


def get_aa_dict() -> Dict[str, str]:
    global aa_dict
    return aa_dict


def get_pdb_id_from_path(path: str) -> str:
    """
    :param path: a pdb file path
    :return: pdb id, using pdb file name as pdb id
    """
    basename = os.path.basename(path)
    return basename.split('.')[0]


def get_pdb(pdb_id: str, path: str) -> Bio.PDB.Structure.Structure:
    """
    :param pdb_id: id in protein data bank, or other self-defined string
    :param path: .pdb file path
    :return: Bio.PDB.Structure.Structure object
    """
    parser = Bio.PDB.PDBParser()
    structure = parser.get_structure(pdb_id, path)
    return structure


def get_chain(pdb_id: str, path: str) -> Bio.PDB.Chain.Chain:
    structure = get_pdb(pdb_id, path)
    model = list(structure.get_models())[0]
    chain = list(model.get_chains())[0]
    return chain


def get_sequence(path: str) -> str:
    """
    :param path: ".fasta" file path that only contains one sequence
    :return: sequence string
    """
    with open(path) as f:
        seqs = list(SimpleFastaParser(f))
    return seqs[0][1]
