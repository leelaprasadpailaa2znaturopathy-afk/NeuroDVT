import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureNormalization(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.theta = nn.Parameter(torch.ones(dim))
        self.lambda_param = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # x shape: (batch, dim)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return x_norm * self.theta + self.lambda_param

class DendriticLayer(nn.Module):
    """
    Bio-inspired Dendritic Layer as specified:
    1. Synapse Layer (Weights and biases)
    2. Dendrite Layer (Normalization and aggregation)
    3. Soma Layer (Final aggregation)
    """
    def __init__(self, input_dim, num_classes, num_branches=16):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_branches = num_branches
        
        # Synapse weights: (num_classes, num_branches, input_dim)
        self.w = nn.Parameter(torch.randn(num_classes, num_branches, input_dim) * 0.01)
        self.b = nn.Parameter(torch.zeros(num_classes, num_branches, input_dim))
        
        self.feature_norm = FeatureNormalization(input_dim)
        self.branch_norm = FeatureNormalization(input_dim)
        
    def forward(self, x):
        # x: (batch, input_dim)
        batch_size = x.size(0)
        
        # Initial feature normalization η(x)
        x_eta = self.feature_norm(x) # (batch, input_dim)
        
        # We need to compute yk = Σ_i Σ_j δ(η(w_kij * η(x)_j + b_kij))
        # This can be vectorized. 
        # x_eta is (batch, input_dim)
        # We want to multiply by w_kij (num_classes, num_branches, input_dim)
        
        # Expand x_eta for broadcasting: (batch, 1, 1, input_dim)
        x_expanded = x_eta.unsqueeze(1).unsqueeze(1) 
        
        # weights w: (1, num_classes, num_branches, input_dim)
        w_expanded = self.w.unsqueeze(0)
        b_expanded = self.b.unsqueeze(0)
        
        # Synapse output: (batch, num_classes, num_branches, input_dim)
        synapse_out = w_expanded * x_expanded + b_expanded
        
        # Apply normalization η to the synapse output
        # Normalization is applied per feature usually, but here we treat input_dim as the feature set
        # Reshape to apply FeatureNormalization easily: (batch * num_classes * num_branches, input_dim)
        flat_synapse = synapse_out.view(-1, self.input_dim)
        norm_synapse = self.branch_norm(flat_synapse)
        
        # Activation function δ (sigmoid or relu, user used δ, usually sigmoid in dendritic models)
        activated = torch.sigmoid(norm_synapse)
        
        # Reconstruct shape: (batch, num_classes, num_branches, input_dim)
        activated = activated.view(batch_size, self.num_classes, self.num_branches, self.input_dim)
        
        # Soma Layer: Sum over branches (i) and input dimensions (j)
        # yk: (batch, num_classes)
        out = activated.sum(dim=(2, 3))
        
        return out

class DendriticClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_branches=16):
        super().__init__()
        self.dendritic = DendriticLayer(input_dim, num_classes, num_branches)
        
    def forward(self, x):
        # x: (batch, input_dim)
        logits = self.dendritic(x)
        return logits
