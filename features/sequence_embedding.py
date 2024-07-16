import torch
import esm
from typing import *
from utils.data import Protein, Dataset


class ResidueEmbeddings:
    def __init__(self, representations: torch.Tensor):
        self.representations = representations
        self.sequence_length = len(representations) - 2

    def get_single_site_residue_embedding_window(self, site_num: int) -> torch.Tensor:
        """
        Generate sequence windows for one residue. Add zero paddings if necessary.
        :param site_num: PTM site sequence number
        :return: 11×1280 sequence window for one PTM site
        """

        sequence_length = self.representations.shape[1] - 2
        token_indexes = [i if 0 < i <= sequence_length else 0 for i in range(site_num - 5, site_num + 5 + 1)]
        # the token with index 0 is a <bos> token
        return torch.stack([self.representations[0, i] if i else torch.zeros(1280) for i in token_indexes])

    def get_residue_embedding_window(self, site1_num: int, site2_num: int) -> torch.Tensor:
        """
        :param site1_num: PTM site1 sequence number
        :param site2_num: PTM site2 sequence number
        :return:
        """
        window1 = self.get_single_site_residue_embedding_window(site1_num)
        window2 = self.get_single_site_residue_embedding_window(site2_num)
        return torch.cat((window1, window2), dim=0)

    def __getitem__(self, item: Union[int, Tuple[int, int]]):
        if isinstance(item, int):
            return self.get_single_site_residue_embedding_window(item)
        elif isinstance(item, tuple):
            return self.get_residue_embedding_window(*item)


class ResiduePairEmbeddings:
    def __init__(self, attentions: torch.Tensor):
        self.attentions = attentions

    def get_residue_pair_embedding(self, site1_num: int, site2_num: int) -> torch.Tensor:
        return torch.cat(
            (
                self.attentions[0, :, :, site1_num, site2_num].reshape(660),
                self.attentions[0, :, :, site2_num, site1_num].reshape(660)
            )
        )

    def __getitem__(self, item: Tuple[int, int]):
        return self.get_residue_pair_embedding(*item)


class SequenceEmbeddingModel:
    """
    A wrap for protein language model ESM-2 (650M), use to generate residue and residue-pair embeddings for each
    PTM pairs in a protein.
    """

    def __init__(self, esm2_650_weights_path):
        self.esm2_model, self.alphabet = esm.pretrained.load_model_and_alphabet_local(esm2_650_weights_path)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.esm2_model.eval()

    def generate_single_protein_embeddings(self, protein: Protein) -> Tuple[ResidueEmbeddings, ResiduePairEmbeddings]:
        batch_labels, batch_strs, batch_tokens = self.batch_converter([(protein.pdb_id, protein.get_sequence())])
        with torch.no_grad():
            results = self.esm2_model(batch_tokens, repr_layers=[33], need_head_weights=True)

        token_representations = results["representations"][33]
        attentions = results["attentions"]
        return ResidueEmbeddings(token_representations), ResiduePairEmbeddings(attentions)

    def __call__(self, dataset: Dataset):
        batched_residue_embedding_windows = []
        batched_residue_pair_embeddings = []

        for protein in dataset:
            residue_embeddings, residue_pair_embeddings = self.generate_single_protein_embeddings(protein)
            for pair in protein:
                batched_residue_embedding_windows.append(residue_embeddings[pair])
                batched_residue_pair_embeddings.append(residue_pair_embeddings[pair])

        return torch.stack(batched_residue_embedding_windows), torch.stack(batched_residue_pair_embeddings)
