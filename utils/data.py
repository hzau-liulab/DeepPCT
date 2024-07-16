import Bio.PDB.Residue
from utils.distance import ProteinResidueDistanceDict, ProteinResidueSpatialNeighbourDict, ProteinGraph
from utils.functional import get_sequence, get_chain, get_pdb_id_from_path, get_aa_dict
from rdkit import Chem
from typing import *


class Protein:
    def __init__(self, fasta_path: str, pdb_path: str, samples: List[Tuple[int, int]]):
        self.fasta_path = fasta_path
        self.pdb_path = pdb_path
        self.pdb_id = get_pdb_id_from_path(self.pdb_path)
        self.sequence = get_sequence(self.fasta_path)
        self.chain = get_chain(self.pdb_id, self.pdb_path)
        self.residues: List[Bio.PDB.Residue.Residue] = list(self.chain)
        self.residue_distance_dict = ProteinResidueDistanceDict(self.pdb_id, self.residues)
        self.cache = {}
        self.neighbour_dicts = {}
        self._generate_rdkit_molecule()
        self.graphs: Dict[Union[int, float], Dict[str, Union[ProteinGraph, None]]] = {}
        # {cutoff: {'residue': ProteinGraph ,'atom': ProteinGraph}}
        self.samples = samples
        self._generate_site_list()
        self._check()

    def _check(self):
        aa_dict = get_aa_dict()

        assert len(self.sequence) == len(self.residues), \
            f'Input sequence file ({self.fasta_path}) and structure file ({self.pdb_path}) do not match in sequence ' \
            f'length. '

        for aa, residue in zip(self.sequence, self.residues):
            assert aa == aa_dict[residue.resname], f'Input sequence file ({self.fasta_path}) and structure file {self.pdb_path} do not match in residue number {residue.id[1]}: {aa} != {aa_dict[residue.resname]}'

    def _generate_site_list(self):
        sites = set()
        for site1, site2 in self.samples:
            sites.add(site1)
            sites.add(site2)
        self.sites: List[int] = list(sites)
        self.sites.sort()

        self.site_residues: List[Bio.PDB.Residue.Residue] = [self.residues[num - 1] for num in self.sites]

    def _generate_rdkit_molecule(self):
        try:
            mol = Chem.MolFromPDBFile(self.pdb_path)
        except ValueError:
            Warning(
                f'RDKit cannot parse the PDB file {self.pdb_path} because there might be two atoms in the PDB file '
                f'that are too close. This can lead RDKit to automatically form a bond between them, resulting in a '
                f'chemically unreasonable structure (e.g., valence of atom is incorrect). '
                f'Proximity bonding in RDKit is currently disabled. '
                f'Trying to parse the PDB file again.'
            )
            mol = Chem.MolFromPDBFile(self.pdb_path, proximityBonding=False)

            print(f'PDB file {self.pdb_path} has successfully parsed.')
        self.rdkit_molecule = mol

    def _generate_protein_graph(self, cutoff: Union[int, float], level: str):
        if cutoff in self.graphs and self.graphs[cutoff][level] is not None:
            return
        self.graphs[cutoff] = {'atom': None, 'residue': None}
        self.graphs[cutoff][level] = ProteinGraph(self.pdb_id, self.residues, cutoff, level)

    def _generate_neighbor_dict(self, num_neighbors: int):
        if num_neighbors in self.neighbour_dicts:
            return

        self.neighbour_dicts[num_neighbors] = ProteinResidueSpatialNeighbourDict(
            self.pdb_id, self.residues, num_neighbors
        )

    def get_sequence(self):
        return self.sequence

    def get_rdkit_molecule(self):
        return self.rdkit_molecule

    def get_residue_distance_dict(self):
        return self.residue_distance_dict

    def get_neighbour_dict(self, num_neighbors: int):
        self._generate_neighbor_dict(num_neighbors)

        return self.neighbour_dicts[num_neighbors]

    def to_networkx(self, cutoff: Union[int, float], level: str):
        self._generate_protein_graph(cutoff, level)
        return self.graphs[cutoff][level].to_networkx()

    def __len__(self):
        return len(self.sequence)

    def __str__(self):
        return f'Protein(PDB={self.pdb_id}, sequence={self.sequence})'

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.__str__() + self.pdb_path)

    def __iter__(self):
        return iter(self.samples)


class Dataset:
    def __init__(self, data: OrderedDict[Tuple[str, str], List[Tuple[int, int]]]):
        """
        :param data: {(fasta_path, pdb_path): [(site1, site2), ...], ...}
        """
        self.proteins: List[Protein] = []
        for (fasta_path, pdb_path), pairs in data.items():
            self.proteins.append(Protein(fasta_path, pdb_path, pairs))

    def __getitem__(self, index):
        return self.proteins[index]

    def __iter__(self):
        return iter(self.proteins)


if __name__ == '__main__':
    p = Protein('../input/sequence/P48431.fasta', '../input/structure/P48431.pdb', [(1, 2)])
