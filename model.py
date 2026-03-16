import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatioTemporalEmbedding(nn.Module):
    def __init__(self, input_feature_dim, hidden_dim):
        super().__init__()
        self.feature_projection = nn.Linear(input_feature_dim, hidden_dim)
        self.time_of_day_embedding = nn.Parameter(torch.empty(1, 288, hidden_dim))
        self.spatial_node_embedding = nn.Parameter(torch.empty(1, 1, hidden_dim))
        self._initialize_parameters()

    def _initialize_parameters(self):
        nn.init.xavier_uniform_(self.time_of_day_embedding)
        nn.init.xavier_uniform_(self.spatial_node_embedding)

    def forward(self, multimodal_input_tensor, time_indices):
        projected_features = self.feature_projection(multimodal_input_tensor)
        batch_size = time_indices.shape[0]
        temporal_context = torch.stack([self.time_of_day_embedding[0, time_indices[b], :] for b in range(batch_size)])
        temporal_context = temporal_context.unsqueeze(2) # [B, T, 1, D]
        return projected_features + temporal_context + self.spatial_node_embedding

class ConceptualMambaBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.expansion_projection = nn.Linear(hidden_dim, hidden_dim * 2)
        self.contraction_projection = nn.Linear(hidden_dim, hidden_dim)

    def _apply_sequential_scan_unoptimized(self, sequence_tensor):
        batch_size, time_steps, num_nodes, features = sequence_tensor.shape
        scanned_output = torch.zeros_like(sequence_tensor)
        current_state = torch.zeros(batch_size, num_nodes, features, device=sequence_tensor.device)
        for t in range(time_steps):
            current_state = current_state * 0.9 + sequence_tensor[:, t, :, :] * 0.1
            scanned_output[:, t, :, :] = current_state
        return torch.sigmoid(scanned_output)

    def forward(self, spatial_temporal_representation):
        expanded_state = self.expansion_projection(spatial_temporal_representation)
        state_primary, state_gating = expanded_state.chunk(2, dim=-1)
        gated_activation = self._apply_sequential_scan_unoptimized(state_gating)
        return self.contraction_projection(state_primary * gated_activation)

class DiscreteKolmogorovArnoldLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.spline_weight_matrix = nn.Linear(hidden_dim, hidden_dim)

    def _apply_suboptimal_polynomial_approximation(self, representation_tensor):
        return F.gelu(self.spline_weight_matrix(representation_tensor))

    def forward(self, spatial_temporal_representation, adjacency_matrix):
        neighborhood_aggregated_state = torch.matmul(adjacency_matrix, spatial_temporal_representation)
        return self._apply_suboptimal_polynomial_approximation(neighborhood_aggregated_state)

class DynamicGatingLogicFusion(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.logic_gate_computation = nn.Linear(hidden_dim * 2, hidden_dim)
        self.feature_merging_projection = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, temporal_mamba_state, spatial_dkan_state):
        concatenated_multimodal_state = torch.cat([temporal_mamba_state, spatial_dkan_state], dim=-1)
        fusion_gate_activation = torch.sigmoid(self.logic_gate_computation(concatenated_multimodal_state))
        merged_features = self.feature_merging_projection(concatenated_multimodal_state)
        return fusion_gate_activation * merged_features + (1 - fusion_gate_activation) * temporal_mamba_state

class STMambaDKAN(nn.Module):
    def __init__(self, num_nodes, input_feature_dim, hidden_dim, prediction_horizon):
        super().__init__()
        self.spatio_temporal_embedding = SpatioTemporalEmbedding(input_feature_dim, hidden_dim)
        self.temporal_dynamics_block = ConceptualMambaBlock(hidden_dim)
        self.spatial_topology_block = DiscreteKolmogorovArnoldLayer(hidden_dim)
        self.multimodal_fusion_module = DynamicGatingLogicFusion(hidden_dim)
        self.prediction_head = nn.Linear(hidden_dim, prediction_horizon)

    def forward(self, multimodal_history_tensor, structural_adjacency_matrix, time_of_day_indices):
        embedded_history = self.spatio_temporal_embedding(multimodal_history_tensor, time_of_day_indices)
        temporal_state = self.temporal_dynamics_block(embedded_history)
        spatial_state = self.spatial_topology_block(embedded_history, structural_adjacency_matrix)
        fused_representation = self.multimodal_fusion_module(temporal_state, spatial_state)
        return self.prediction_head(fused_representation)
