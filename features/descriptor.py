import os
import itertools
import tempfile
import subprocess
import numpy as np
import networkx as nx
import Bio.PDB.Residue
import Bio.PDB.Atom
import utils.functional as func
from utils.data import Protein, Dataset
from scipy import stats
from typing import *
from abc import ABCMeta, abstractmethod
from GraphRicciCurvature.OllivierRicci import OllivierRicci


class Feature(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, dataset: Dataset) -> np.ndarray:
        pass


class ResidueFeature(Feature):
    @abstractmethod
    def __call__(self, dataset: Dataset) -> np.ndarray:
        pass

    @staticmethod
    def generate_feature_vector(site1_value: Union[int, float], site2_value: Union[int, float]) -> List[
        Union[int, float]
    ]:
        return [min(site1_value, site2_value), max(site1_value, site2_value), (site1_value + site2_value) / 2]


class ShortestPathDistance(Feature):
    def __init__(self):
        super(ShortestPathDistance, self).__init__()

    def __call__(self, dataset: Dataset) -> np.ndarray:
        feature_vectors = []
        for protein in dataset:
            protein_graph = protein.to_networkx(cutoff=7, level='residue')
            for site1, site2 in protein:
                feature_vector = [nx.shortest_path_length(G=protein_graph, source=site1, target=site2)]
                feature_vectors.append(feature_vector)
        return np.array(feature_vectors)


class CircularVariance(ResidueFeature):
    def __init__(self):
        super(CircularVariance, self).__init__()

    @staticmethod
    def get_surrounding_atoms(
            atom_i: Bio.PDB.Atom.Atom,
            residues: List[Bio.PDB.Residue.Residue]
    ) -> List[Bio.PDB.Atom.Atom]:
        surrounding_atoms = []
        for residue in residues:
            for atom_j in residue.get_atoms():
                if atom_i != atom_j:
                    if np.linalg.norm(atom_i.get_coord() - atom_j.get_coord()) <= 10:
                        surrounding_atoms.append(atom_j)
        return surrounding_atoms

    def cal_atom_circular_variance(self, atom_i: Bio.PDB.Atom.Atom, residues: List[Bio.PDB.Residue.Residue]) -> float:
        vector_i = atom_i.get_coord()
        r_ij_ele_list = []
        surrounding_atoms = self.get_surrounding_atoms(atom_i, residues)
        for atom_j in surrounding_atoms:
            vector_j = atom_j.get_coord()
            r_ij = vector_j - vector_i
            r_ij_ele_list.append(r_ij / np.linalg.norm(r_ij))
        sum_r_ij = np.array(r_ij_ele_list).sum(axis=0)
        n = len(r_ij_ele_list)
        cv_i = 1 - np.linalg.norm(sum_r_ij) / n
        return cv_i

    def cal_circular_variance(self, protein: Protein) -> Dict[int, float]:
        """
        :return: {structure number: circular variance}
        """
        residues: List[Bio.PDB.Residue.Residue] = protein.residues
        cv_dict = {}
        for residue_i in protein.site_residues:
            structure_number_i = residue_i.id[1]
            residue_cv_values = []
            for atom_i in residue_i.get_atoms():
                cv_i = self.cal_atom_circular_variance(atom_i, residues)
                residue_cv_values.append(cv_i)
            cv_dict[structure_number_i] = float(sum(residue_cv_values) / len(residue_cv_values))
        return cv_dict

    def __call__(self, dataset: Dataset) -> np.ndarray:
        feature_vectors = []
        for protein in dataset:
            cv_dict = self.cal_circular_variance(protein)
            for site1, site2 in protein:
                feature_vectors.append(self.generate_feature_vector(cv_dict[site1], cv_dict[site2]))
        return np.array(feature_vectors)


class GhecomCalculatedFeature(ResidueFeature):
    def __init__(self):
        super(GhecomCalculatedFeature, self).__init__()

    @staticmethod
    def _run_ghecom(protein: Protein):
        temp_file = tempfile.NamedTemporaryFile(mode='w+t', delete=False)
        temp_file.close()
        program_path = f'{func.get_project_root()}/software/ghecom/ghecom'
        args = [program_path, '-M', 'M', '-ipdb', protein.pdb_path, '-ores', temp_file.name,
                '-clus', 'T', '-rlx', '10.0', '-sprb', 'F']

        subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        with open(temp_file.name, 'r') as f:
            lines = f.read().splitlines()
        os.remove(temp_file.name)

        shell_accessibility_dict = {}
        minimum_inaccessible_radius_dict = {}
        pocketness_dict = {}

        for line in filter(lambda x: not x.startswith('#'), lines):
            line_elements = line.split()

            structure_number = int(line_elements[0])
            shell_accessibility = float(line_elements[3])
            minimum_inaccessible_radius = float(line_elements[4])
            pocketness = float(line_elements[7])

            shell_accessibility_dict[structure_number] = shell_accessibility
            minimum_inaccessible_radius_dict[structure_number] = minimum_inaccessible_radius
            pocketness_dict[structure_number] = pocketness

        protein.cache['shell_accessibility_dict'] = shell_accessibility_dict
        protein.cache['minimum_inaccessible_radius_dict'] = minimum_inaccessible_radius_dict
        protein.cache['pocketness_dict'] = pocketness_dict

    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class AccessibleShellVolume(GhecomCalculatedFeature):

    def __init__(self):
        super(AccessibleShellVolume, self).__init__()

    def _cal_accessible_shell_volume(self, protein: Protein):
        if protein.cache.get('shell_accessibility_dict') is None:
            self._run_ghecom(protein)

        return protein.cache['shell_accessibility_dict']

    def __call__(self, dataset: Dataset) -> np.ndarray:
        feature_vectors = []
        for protein in dataset:
            shell_accessibility_dict = self._cal_accessible_shell_volume(protein)
            for site1, site2 in protein:
                value1 = shell_accessibility_dict[site1]
                value2 = shell_accessibility_dict[site2]
                feature_vectors.append(self.generate_feature_vector(value1, value2))
        return np.array(feature_vectors)


class MinimumInaccessibleRadius(GhecomCalculatedFeature):
    def __init__(self):
        super(MinimumInaccessibleRadius, self).__init__()

    def _cal_minimum_inaccessible_radius(self, protein: Protein):
        if protein.cache.get('minimum_inaccessible_radius_dict') is None:
            self._run_ghecom(protein)

        return protein.cache['minimum_inaccessible_radius_dict']

    def __call__(self, dataset: Dataset) -> np.ndarray:
        feature_vectors = []
        for protein in dataset:
            minimum_inaccessible_radius_dict = self._cal_minimum_inaccessible_radius(protein)
            for site1, site2 in protein:
                value1 = minimum_inaccessible_radius_dict[site1]
                value2 = minimum_inaccessible_radius_dict[site2]
                feature_vectors.append(self.generate_feature_vector(value1, value2))
        return np.array(feature_vectors)


class Pocketness(GhecomCalculatedFeature):
    def __init__(self):
        super(Pocketness, self).__init__()

    def _cal_pocketness(self, protein: Protein):
        if protein.cache.get('pocketness_dict') is None:
            self._run_ghecom(protein)

        return protein.cache['pocketness_dict']

    def __call__(self, dataset: Dataset) -> np.ndarray:
        feature_vectors = []
        for protein in dataset:
            pocketness_dict = self._cal_pocketness(protein)
            for site1, site2 in protein:
                value1 = pocketness_dict[site1]
                value2 = pocketness_dict[site2]
                feature_vectors.append(self.generate_feature_vector(value1, value2))
        return np.array(feature_vectors)


class MultiFractalDimension(ResidueFeature):
    def __init__(self):
        super(MultiFractalDimension, self).__init__()

    @staticmethod
    def _cal_mfd(protein: Protein, weight=None):
        mfd_dict = {}
        graph = protein.to_networkx(cutoff=7, level='residue')
        for node in protein.sites:
            grow = []
            r_g = []
            num_g = []
            num_nodes = 0
            if weight is None:
                spl = nx.single_source_shortest_path_length(graph, node)
            else:
                spl = nx.single_source_dijkstra_path_length(graph, node)
            for s in spl.values():
                if s > 0:
                    grow.append(s)
            grow.sort()
            num = Counter(grow)
            for i, j in num.items():
                num_nodes += j
                if i > 0:
                    r_g.append(i)
                    num_g.append(num_nodes)
            x = np.log(r_g)
            y = np.log(num_g)
            if len(r_g) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                mfd_dict[node] = slope
            else:
                mfd_dict[node] = 0
        return mfd_dict

    def __call__(self, dataset: Dataset) -> np.ndarray:
        feature_vectors = []
        for protein in dataset:
            mdf_dict = self._cal_mfd(protein)
            for site1, site2 in protein:
                value1 = mdf_dict[site1]
                value2 = mdf_dict[site2]
                feature_vectors.append(self.generate_feature_vector(value1, value2))
        return np.array(feature_vectors)


class OsipovPickupDunmurChiralityIndex(ResidueFeature):
    def __init__(self):
        super(OsipovPickupDunmurChiralityIndex, self).__init__()
        self.N = (5, 7, 10, 15)
        self.molecular_weight = {
            'A': 89.1,
            'R': 174.2,
            'N': 132.1,
            'D': 133.1,
            'C': 121.2,
            'E': 147.1,
            'Q': 146.2,
            'G': 75.1,
            'H': 155.2,
            'I': 131.2,
            'L': 131.2,
            'K': 146.2,
            'M': 149.2,
            'F': 165.2,
            'P': 115.1,
            'S': 105.1,
            'T': 119.1,
            'W': 204.2,
            'Y': 181.2,
            'V': 117.1
        }

    @staticmethod
    def _osipov(protein: Protein, coord: List[np.ndarray], molecular_weights: List[np.ndarray], num_neighbors: int) -> \
            Dict[int, float]:
        G_os = {}
        sites_idx = [i - 1 for i in protein.sites]
        for idx in sites_idx:
            G = 0
            for P in itertools.permutations(np.arange(num_neighbors), 4):
                r_ij = coord[idx][P[0]] - coord[idx][P[1]]
                r_kl = coord[idx][P[2]] - coord[idx][P[3]]
                r_il = coord[idx][P[0]] - coord[idx][P[3]]
                r_jk = coord[idx][P[1]] - coord[idx][P[2]]

                r_ij_mag = np.linalg.norm(r_ij)
                r_kl_mag = np.linalg.norm(r_kl)
                r_il_mag = np.linalg.norm(r_il)
                r_jk_mag = np.linalg.norm(r_jk)

                mw = molecular_weights[idx][P[0]] * molecular_weights[idx][P[1]] * molecular_weights[idx][P[2]] * \
                     molecular_weights[idx][P[3]]

                G_p_up = np.dot(np.cross(r_ij, r_kl), r_il) * (np.dot(r_ij, r_jk)) * (
                    np.dot(r_jk, r_kl))
                G_p_down = ((r_ij_mag * r_jk_mag * r_kl_mag) ** 2) * r_il_mag
                G_p = mw * G_p_up / G_p_down

                G += G_p
            G_os[idx + 1] = ((4 * 3 * 2 * 1) / (num_neighbors ** 4) * (1 / 3) * G)

        return G_os

    def _cal_coord_list_and_molecular_weight_list(self, protein: Protein, num_neighbors):
        sequence = protein.get_sequence()
        neighbor_dict = protein.get_neighbour_dict(num_neighbors=num_neighbors - 1)
        coord_list, molecular_weight_list = [], []
        for idx, aa in enumerate(sequence):
            num = idx + 1
            neighbors = neighbor_dict[num]

            nei_coord_list, nei_weight_list = [protein.residues[num - 1].child_dict['CA'].get_coord()], [
                self.molecular_weight[aa]]
            for nei_num in neighbors:
                nei_aa = sequence[nei_num - 1]
                coord = protein.residues[nei_num - 1].child_dict['CA'].get_coord()
                nei_coord_list.append(coord)
                nei_weight_list.append(self.molecular_weight[nei_aa])
            coord_list.append(np.array(nei_coord_list))
            molecular_weight_list.append(np.array(nei_weight_list))
        return coord_list, molecular_weight_list

    def cal_opd_index(self, protein: Protein) -> Dict[int, List[float]]:
        OPD_dict = {}
        for num_neighbors in self.N:
            coord_list, molecular_weight_list = self._cal_coord_list_and_molecular_weight_list(protein, num_neighbors)
            G_os = self._osipov(protein, coord_list, molecular_weight_list, num_neighbors)
            for residue_num, opd_value in G_os.items():
                if residue_num in OPD_dict:
                    OPD_dict[residue_num].append(opd_value)
                else:
                    OPD_dict[residue_num] = [opd_value]
        return OPD_dict

    def __call__(self, dataset: Dataset) -> np.ndarray:
        feature_vectors = []
        for protein in dataset:
            opd_dict = self.cal_opd_index(protein)
            for site1, site2 in protein:
                feature_vector = []
                for index in range(len(self.N)):
                    value1, value2 = opd_dict[site1][index], opd_dict[site2][index]
                    feature_vector.extend(self.generate_feature_vector(value1, value2))
                feature_vectors.append(feature_vector)
        return np.array(feature_vectors)


class OllivierRicciCurvature(ResidueFeature):
    """
    Can not execute on Windows: fork method is not supported on Windows and spawn is not supported for this package.
    """

    def __init__(self):
        super(OllivierRicciCurvature, self).__init__()

    @staticmethod
    def _cal_ollivier_ricci(graph):
        orc = OllivierRicci(graph)
        orc.compute_ricci_curvature()
        orc_dict = {}
        for node_num in graph.nodes():
            orc_dict[node_num] = sum(nei['ricciCurvature'] for nei in orc.G[node_num].values())
        return orc_dict

    def __call__(self, dataset: Dataset) -> np.ndarray:
        feature_vectors = []
        for protein in dataset:
            orc_dict = self._cal_ollivier_ricci(protein.to_networkx(cutoff=7, level='residue'))
            for site1, site2 in protein:
                value1 = orc_dict[site1]
                value2 = orc_dict[site2]
                feature_vectors.append(self.generate_feature_vector(value1, value2))
        return np.array(feature_vectors)


class DescriptorsCalculator:

    def __init__(self):
        self.calculators = [
            ShortestPathDistance(), CircularVariance(), AccessibleShellVolume(),
            MinimumInaccessibleRadius(), Pocketness(), MultiFractalDimension(),
            OsipovPickupDunmurChiralityIndex(), OllivierRicciCurvature()
        ]

    def __call__(self, dataset: Dataset) -> np.ndarray:
        results = [calculator(dataset) for calculator in self.calculators]
        return np.concatenate(results, axis=1)
