import torch
from torchdrug import layers, models, data
from torchdrug.layers import geometry
from utils.data import Protein


class StructuralResidueEmbeddings:
    def __init__(self, representations: torch.Tensor):
        self.representations = representations

    def __getitem__(self, site_num: int):
        return self.representations[site_num - 1]


class StructuralEmbeddingModel:
    """
    A wrap for GearNet-Edge model
    """

    def __init__(self, gearnet_edge_weights_path: str):
        self._init_gearnet_edge_model(gearnet_edge_weights_path)
        self._init_gearnet_edge_graph_construction_model()

    def _init_gearnet_edge_model(self, gearnet_edge_weights_path):
        self.gearnet_edge_model = models.GearNet(
            input_dim=21, hidden_dims=[512, 512, 512, 512, 512, 512],
            num_relation=7, edge_input_dim=59, num_angle_bin=8,
            batch_norm=True, concat_hidden=True, short_cut=True, readout="sum"
        )

        self.gearnet_edge_model.load_state_dict(
            torch.load(gearnet_edge_weights_path, map_location=torch.device('cpu'))
        )

        self.gearnet_edge_model.eval()

    def _init_gearnet_edge_graph_construction_model(self):
        self.gearnet_edge_graph_construction_model = layers.GraphConstruction(
            node_layers=[geometry.AlphaCarbonNode()],
            edge_layers=[
                geometry.SequentialEdge(max_distance=2),
                geometry.SpatialEdge(radius=10.0, min_distance=5),
                geometry.KNNEdge(k=10, min_distance=5),
            ],
            edge_feature="gearnet"
        )

    def construct_gearnet_edge_graph(self, mol):
        protein = data.Protein.from_molecule(
            mol=mol,
            atom_feature="position",
            bond_feature="length",
            residue_feature="symbol"
        )
        protein.view = 'residue'
        protein = data.Protein.pack([protein])
        protein_graph = self.gearnet_edge_graph_construction_model(protein)
        return protein_graph

    def __call__(self, protein: Protein):
        protein_graph = self.construct_gearnet_edge_graph(protein.get_rdkit_molecule())
        with torch.no_grad():
            representations = self.gearnet_edge_model(protein_graph, protein_graph.node_feature.float())

        return StructuralResidueEmbeddings(representations['node_feature'])
