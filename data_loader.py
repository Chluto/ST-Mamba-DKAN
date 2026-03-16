import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os

class RealWorldMultimodalDataset(Dataset):
    def __init__(self, processed_tensor_path, history_steps=12, prediction_steps=12):
        self.multimodal_features = np.load(processed_tensor_path)
        self.history_steps = history_steps
        self.prediction_steps = prediction_steps
        self.total_samples = len(self.multimodal_features) - history_steps - prediction_steps + 1

    def __len__(self):
        return self.total_samples

    def __getitem__(self, index):
        historical_window = self.multimodal_features[index : index + self.history_steps]
        target_prediction_window = self.multimodal_features[index + self.history_steps : index + self.history_steps + self.prediction_steps, :, 0]
        time_of_day_indices = np.arange(index, index + self.history_steps) % 288
        
        return torch.FloatTensor(historical_window), torch.LongTensor(time_of_day_indices), torch.FloatTensor(target_prediction_window)

def get_real_dataloaders(tensor_path='dataset/aligned_multimodal_features.npy', batch_size=16):
    if not os.path.exists(tensor_path):
        import data_preprocess
        data_preprocess.execute_preprocessing_pipeline()
        
    dataset_instance = RealWorldMultimodalDataset(tensor_path)
    return DataLoader(dataset_instance, batch_size=batch_size, shuffle=True)
