import Bio.PDB.Residue
import Bio.PDB.Atom
import numpy as np
import networkx as nx
from typing import *


class ProteinResidueDistanceDict:
    def __init__(self, pdb_id, residues: List[Bio.PDB.Residue.Residue]):
        self.pdb_id = pdb_id
        self.residues = residues
        self._calculate_distance_dict()

    def _calculate_distance_dict(self):
        self.distance_dict = {}
        for i, residue_i in enumerate(self.residues):
            alpha_C_i_coord: np.ndarray = residue_i.child_dict['CA'].get_coord()
            structure_number_i = residue_i.id[1]
            for j in range(i + 1, len(self.residues)):
                residue_j = self.residues[j]
                alpha_C_j_coord = residue_j.child_dict['CA'].get_coord()
                structure_number_j = residue_j.id[1]
                alphaC_coord_diff = alpha_C_i_coord - alpha_C_j_coord
                distance = np.linalg.norm(alphaC_coord_diff)
                self.distance_dict[(structure_number_i, structure_number_j)] = distance
                self.distance_dict[(structure_number_j, structure_number_i)] = distance

    def _convert_dict(self):
        """
         {structure number1: {structure number2: distance}}
        """
        if hasattr(self, 'converted_distance_dict'):
            return
        converted_dict = {}
        for (structure_number1, structure_number2), distance in self.distance_dict.items():
            if structure_number1 in converted_dict:
                converted_dict[structure_number1][structure_number2] = distance
            else:
                converted_dict[structure_number1] = {structure_number2: distance}
        self.converted_distance_dict: Dict[int, Dict[int, float]] = converted_dict

    def __getitem__(self, item: Union[Tuple[int, int], int]) -> Union[float, Dict[int, float]]:
        """
        :param item: (structure number1, structure number2) or structure number1
        :return: distance or {structure number2: distance2, ...}
        """
        if isinstance(item, tuple):
            return self.distance_dict[item]
        elif isinstance(item, int):
            self._convert_dict()
            return self.converted_distance_dict[item]
        else:
            raise ValueError(f'Unsupported key: {item}')

    def __str__(self):
        return f'ProteinDistanceDict(PDB: {self.pdb_id}, Dict: {self.distance_dict})'

    def __repr__(self):
        return self.__str__()


class ProteinResidueSpatialNeighbourDict(ProteinResidueDistanceDict):
    def __init__(self, pdb_id: str, residues: List[Bio.PDB.Residue.Residue], num_neighbors: int):
        super(ProteinResidueSpatialNeighbourDict, self).__init__(pdb_id, residues)
        self._cal_neighbors(num_neighbors)

    def _cal_neighbors(self, num_neighbors: int):
        self._convert_dict()
        neighbor_dict = {}
        for structure_number, neighbor_distance_dict in self.converted_distance_dict.items():
            neighbor_distance = list(neighbor_distance_dict.items())
            neighbor_distance.sort(key=lambda x: x[1])
            neighbors = [neighbor_number for neighbor_number, _ in neighbor_distance[:num_neighbors]]
            neighbor_dict[structure_number] = neighbors
        self.neighbor_dict: Dict[int, List[int]] = neighbor_dict

    def __getitem__(self, item: int):
        return self.neighbor_dict[item]

    def __str__(self):
        return f'ProteinResidueSpatialNeighbourDict(PDB: {self.pdb_id}, Dict: {self.neighbor_dict})'

    def __repr__(self):
        return self.__str__()


class ProteinGraph(ProteinResidueDistanceDict):
    def __init__(self, pdb_id: str, residues: List[Bio.PDB.Residue.Residue], cutoff: Union[int, float],
                 level: str = 'residue'):
        super(ProteinGraph, self).__init__(pdb_id, residues)
        self.cutoff = cutoff
        self.level = level
        self._cal_edges()

    def _cal_edges(self):
        edges = []

        if self.level == 'atom':
            raise NotImplementedError

        elif self.level == 'residue':
            for (structure_number1, structure_number2), distance in self.distance_dict.items():
                if distance <= self.cutoff:
                    edges.append((structure_number1, structure_number2))

        else:
            raise ValueError(f'level {self.level} is not supported.')

        self.edges = edges

    def to_networkx(self) -> nx.Graph:
        G = nx.from_edgelist(self.edges)
        return G

    def __str__(self):
        return f'ProteinGraph(PDB: {self.pdb_id}, Edge: {self.edges})'

    def __repr__(self):
        return self.__str__()

    def __getitem__(self, item):
        raise NotImplementedError
