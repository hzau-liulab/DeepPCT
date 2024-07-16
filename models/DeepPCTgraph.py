import torch
import torch.nn as nn
import numpy as np
from dgl.nn.pytorch import GINConv
from models.GraphConstruction import GraphConstructionModel
from safetensors.torch import load_model
from utils.data import Dataset


class DeepPCTgraph(nn.Module):
    def __init__(self):
        super().__init__()
        self.gin_layer1 = GINConv(nn.Linear(3072, 1544), aggregator_type='max')
        self.gin_layer2 = GINConv(nn.Linear(1544, 722), aggregator_type='max')
        self.layer_norm = nn.LayerNorm(3072)
        self.fc_layer = nn.Sequential(
            nn.Linear(22 * 722, 3971, bias=False),
            # READOUT end
            nn.LeakyReLU(),
            nn.Linear(3971, 2),
        )

    def forward(self, g):
        feat = g.ndata['feat']
        feat = self.layer_norm(feat)
        h = self.gin_layer1(g, feat)
        h = self.gin_layer2(g, h)
        # h: (N * 22) * 722 , N denotes batch size
        # READOUT: node features => graph representation
        h = h.view(-1, 22 * 722)
        # h: N * (22 * 722)
        return self.fc_layer(h)


class DeepPCTgraphInferModel:

    def __init__(self, weight_path: str, gearnet_edge_model_weight_path: str):
        self.graph_construction_model = GraphConstructionModel(gearnet_edge_model_weight_path)
        self._init_model(weight_path)

    def _init_model(self, weight_path: str):
        self.model = DeepPCTgraph()
        load_model(self.model, weight_path)
        self.model.eval()

    def __call__(self, dataset: Dataset) -> np.ndarray:
        """
        :return: [probability1, ... ]
        """
        graph = self.graph_construction_model(dataset)
        with torch.no_grad():
            output = self.model(graph)
        probs = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
        return probs
