import numpy as np
from models.DeepPCTseq import DeepPCTseqInferModel
from models.DeepPCTgraph import DeepPCTgraphInferModel
from models.DeepPCTsite import DeepPCTsiteInferModel
from utils.data import Dataset


class DeepPCTInferModel:
    def __init__(
            self,
            seq_weight_path: str,
            graph_weight_path: str,
            site_weight_path: str,
            esm2_650_weights_path: str,
            gearnet_edge_model_weight_path: str
    ):
        self.seq_model = DeepPCTseqInferModel(seq_weight_path, esm2_650_weights_path)
        self.graph_model = DeepPCTgraphInferModel(graph_weight_path, gearnet_edge_model_weight_path)
        self.site_model = DeepPCTsiteInferModel(site_weight_path)
        self.alpha = 0.65
        self.beta = 0.65

    def __call__(self, dataset: Dataset) -> np.ndarray:
        prob_seq = self.seq_model(dataset)
        prob_graph = self.graph_model(dataset)
        prob_site = self.site_model(dataset)

        return self.beta * prob_seq + (1 - self.beta) * (self.alpha * prob_graph + (1 - self.alpha) * prob_site)
