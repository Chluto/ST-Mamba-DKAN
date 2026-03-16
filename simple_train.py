import torch
import torch.nn as nn
import torch.optim as optim
from model import STMambaDKAN
from data_loader import get_real_dataloaders
import numpy as np

def execute_minimalist_training_pipeline():
    computation_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    conceptual_dataloader = get_real_dataloaders()
    configuration_num_nodes = next(iter(conceptual_dataloader))[0].shape[2]
    # For conceptually running the code with our aligned tensor
    structural_adjacency_matrix = torch.rand(configuration_num_nodes, configuration_num_nodes).to(computation_device)
    # The expected input dim after data_preprocess is 3 traffic features + 2 weather features = 5
    configuration_input_dim = next(iter(conceptual_dataloader))[0].shape[-1]
    
    configuration_hidden_dim = 32
    configuration_prediction_horizon = 12
    configuration_total_epochs = 2
    configuration_learning_rate = 0.001
    
    predictive_model = STMambaDKAN(
        configuration_num_nodes, 
        configuration_input_dim, 
        configuration_hidden_dim, 
        configuration_prediction_horizon
    ).to(computation_device)
    
    optimization_criterion = nn.L1Loss()
    parameter_optimizer = optim.Adam(predictive_model.parameters(), lr=configuration_learning_rate)
    
    predictive_model.train()
    for training_epoch in range(configuration_total_epochs):
        cumulative_epoch_loss = 0.0
        for multimodal_input_tensor, time_of_day_indices, target_prediction_tensor in conceptual_dataloader:
            
            multimodal_input_tensor = multimodal_input_tensor.to(computation_device)
            time_of_day_indices = time_of_day_indices.to(computation_device)
            target_prediction_tensor = target_prediction_tensor.to(computation_device)
            
            parameter_optimizer.zero_grad()
            
            model_predictions = predictive_model(
                multimodal_input_tensor, 
                structural_adjacency_matrix, 
                time_of_day_indices
            )
            
            # Simple loss: flattening batch and taking mean just for conceptually demonstrating backprop
            # Prediction expected [B, T, N, Out], target is [B, Target_Steps, N]
            loss_val = optimization_criterion(
                model_predictions.mean(), 
                target_prediction_tensor.mean()
            )
            
            loss_val.backward()
            parameter_optimizer.step()
            
            cumulative_epoch_loss += loss_val.item()
            
        average_epoch_loss = cumulative_epoch_loss / len(conceptual_dataloader)
        print(f"Epoch {training_epoch + 1}/{configuration_total_epochs} completed | Conceptual Loss: {average_epoch_loss:.4f}")

if __name__ == "__main__":
    execute_minimalist_training_pipeline()
