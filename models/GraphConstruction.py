import dgl
import torch
from features.structural_embedding import StructuralEmbeddingModel
from utils.data import Dataset
from typing import *


class GraphConstructionModel:
    def __init__(self, gearnet_edge_weight_path):
        self.structural_embedding_model = StructuralEmbeddingModel(gearnet_edge_weight_path)

    @staticmethod
    def get_contact_list(nei_num_1: List[int], nei_num_2: List[int]) -> Tuple[List[int], List[int]]:
        contact_list_1 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11]
        contact_list_2 = [11, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        nei_seqNum_graphNum_1 = [(seq_num, index + 1) for index, seq_num in enumerate(nei_num_1)]
        nei_seqNum_graphNum_2 = [(seq_num, index + 12) for index, seq_num in enumerate(nei_num_2)]
        nei_seqNum_graphNum_1.sort(key=lambda x: x[0])
        nei_seqNum_graphNum_2.sort(key=lambda x: x[0])
        for nei_seqNum_graphNum_list in (nei_seqNum_graphNum_1, nei_seqNum_graphNum_2):
            for index, (seq_num, graph_num) in enumerate(nei_seqNum_graphNum_list):
                if index == len(nei_seqNum_graphNum_list) - 1:
                    break
                if seq_num + 1 == nei_seqNum_graphNum_list[index + 1][0]:
                    contact_list_1.append(graph_num)
                    contact_list_2.append(nei_seqNum_graphNum_list[index + 1][1])
        return contact_list_1, contact_list_2

    @staticmethod
    def get_basic_graph(nei_num1: List[int], nei_num2: List[int]) -> dgl.DGLGraph:
        contact_list_1, contact_list_2 = GraphConstructionModel.get_contact_list(nei_num1, nei_num2)
        g = dgl.graph((
            torch.tensor(contact_list_1),
            torch.tensor(contact_list_2)
        ))
        g = dgl.to_bidirected(g)
        return g

    def __call__(self, dataset: Dataset):
        graphs = []
        for protein in dataset:
            neighbour_dict = protein.get_neighbour_dict(num_neighbors=10)
            embeddings = self.structural_embedding_model(protein)
            for site1, site2 in protein:
                neighbours1, neighbours2 = neighbour_dict[site1], neighbour_dict[site2]
                g = self.get_basic_graph(nei_num1=neighbours1, nei_num2=neighbours2)
                feat = [embeddings[site1]]

                for num in neighbours1:
                    feat.append(embeddings[num])

                feat.append(embeddings[site2])
                for num in neighbours2:
                    feat.append(embeddings[num])

                g.ndata['feat'] = torch.stack(feat)
                graphs.append(g)

        return dgl.batch(graphs)
