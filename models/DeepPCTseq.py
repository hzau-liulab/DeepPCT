import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_model
from features.sequence_embedding import SequenceEmbeddingModel
from utils.data import Dataset


class CrossAttentionMutilLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attention_layers = nn.ModuleList(
            nn.MultiheadAttention(embed_dim=1280, num_heads=5, batch_first=True)
            for _ in range(2)
        )
        self.layer_norm = nn.LayerNorm(torch.Size((11, 1280)))

    def forward(self, tgt, memory):
        for layer in self.cross_attention_layers:
            x, _ = layer(query=tgt, key=memory, value=memory, need_weights=False)
            tgt = self.layer_norm(tgt + x)
        return tgt


class DeepPCTseq(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_attention_layer = CrossAttentionMutilLayer()
        self.fnn_layer = nn.Sequential(
            nn.Linear(1280 + 660 * 2, 660),
            nn.BatchNorm1d(660),
            nn.LeakyReLU(),
            nn.Linear(660, 2)
        )

    def forward(self, residue_embedding_window, residue_pair_embedding):
        """
        :param residue_embedding_window: (N, 22, 1280)
        :param residue_pair_embedding: (N, 660 * 2)
        """
        x_window_1, x_window_2 = residue_embedding_window[:, :11, :], residue_embedding_window[:, 11:, :]
        # x_window_i: (N, 11, 1280), i = 1, 2
        x = self.cross_attention_layer(tgt=x_window_2, memory=x_window_1)
        x = torch.cat((x[:, 5, :], residue_pair_embedding), dim=1)
        return self.fnn_layer(x)


class DeepPCTseqInferModel:
    def __init__(self, weight_path: str, esm2_650_weights_path: str):
        super(DeepPCTseqInferModel, self).__init__()
        self._init_model(weight_path, esm2_650_weights_path)

    def _init_embedding_model(self, esm2_650_weights_path: str):
        self.embedding_model = SequenceEmbeddingModel(esm2_650_weights_path)

    def _init_model(self, weight_path, esm2_650_weights_path: str):
        self.model = DeepPCTseq()
        load_model(self.model, weight_path)
        self.model.eval()
        self._init_embedding_model(esm2_650_weights_path)

    def __call__(self, dataset: Dataset) -> np.ndarray:
        """
        :return: [probability1, ... ]
        """
        residue_embedding_windows, residue_pair_embeddings = self.embedding_model(dataset)
        with torch.no_grad():
            output = self.model(residue_embedding_windows, residue_pair_embeddings)
        probs = torch.softmax(output, dim=1)[:, 1].cpu().numpy()
        return probs

