import pickle
import numpy as np
from features.descriptor import DescriptorsCalculator
from utils.data import Dataset


class DeepPCTsiteInferModel:
    def __init__(self, model_path: str):
        self._load_model(model_path)
        self.descriptors_calculator = DescriptorsCalculator()

    def _load_model(self, model_path: str):
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)

    def __call__(self, dataset: Dataset) -> np.ndarray:
        feature_vectors = self.descriptors_calculator(dataset)
        prob = self.model.predict_proba(feature_vectors)[:, 1]
        return prob

