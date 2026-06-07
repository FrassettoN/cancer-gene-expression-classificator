import torch.nn as nn


# Define the MLP model
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)    # First fully connected layer
        self.relu = nn.ReLU()                          # ReLU activation function
        self.fc2 = nn.Linear(hidden_dim, output_dim)   # Final layer for classification

    def forward(self, x):
        x = self.fc1(x)                                # Apply first linear transformation
        x = self.relu(x)                               # Apply ReLU activation
        x = self.fc2(x)                                # Apply final linear transformation (no activation for classification)
        return x

