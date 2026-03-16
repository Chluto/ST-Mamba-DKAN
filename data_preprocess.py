import numpy as np
import pandas as pd

def load_traffic_tensor(file_path):
    compressed_archive = np.load(file_path)
    return compressed_archive['data']

def load_weather_dataframe(file_path):
    return pd.read_csv(file_path, parse_dates=['timestamp'])

def align_timestamps(traffic_tensor, weather_dataframe, expected_timesteps):
    truncated_traffic = traffic_tensor[:expected_timesteps]
    weather_features = weather_dataframe[['temperature', 'precipitation']].values[:expected_timesteps]
    num_nodes = truncated_traffic.shape[1]
    broadcasted_weather = np.tile(weather_features[:, np.newaxis, :], (1, num_nodes, 1))
    return np.concatenate([truncated_traffic, broadcasted_weather], axis=-1)

def impute_missing_values(multimodal_tensor):
    nan_mask = np.isnan(multimodal_tensor)
    multimodal_tensor[nan_mask] = 0.0
    return multimodal_tensor

def normalize_traffic_flow(multimodal_tensor, feature_indices):
    for feature_index in feature_indices:
        feature_slice = multimodal_tensor[..., feature_index]
        mean_value = np.mean(feature_slice)
        standard_deviation = np.std(feature_slice) + 1e-8
        multimodal_tensor[..., feature_index] = (feature_slice - mean_value) / standard_deviation
    return multimodal_tensor

def execute_preprocessing_pipeline():
    traffic_file_path = 'dataset/pemsd4_subset.npz'
    weather_file_path = 'dataset/weather_raw.csv'
    
    traffic_history_tensor = load_traffic_tensor(traffic_file_path)
    weather_context_dataframe = load_weather_dataframe(weather_file_path)
    
    aligned_multimodal_tensor = align_timestamps(
        traffic_history_tensor, 
        weather_context_dataframe, 
        min(len(traffic_history_tensor), len(weather_context_dataframe))
    )
    
    imputed_multimodal_tensor = impute_missing_values(aligned_multimodal_tensor)
    
    normalized_multimodal_tensor = normalize_traffic_flow(imputed_multimodal_tensor, feature_indices=[0, 1])
    
    np.save('dataset/aligned_multimodal_features.npy', normalized_multimodal_tensor)

if __name__ == "__main__":
    execute_preprocessing_pipeline()
