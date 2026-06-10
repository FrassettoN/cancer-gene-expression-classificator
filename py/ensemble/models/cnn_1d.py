import torch
import torch.nn as nn


class CNN_1D(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(CNN_1D, self).__init__()

        self.input_dim = input_dim
        self.num_classes = num_classes
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.feature_dim = self._calculate_feature_dim(input_dim)

        self.fc1 = nn.Linear(self.feature_dim, 128) 
        self.fc2 = nn.Linear(128, num_classes)

    def _calculate_feature_dim(self, input_dim):
        x = torch.zeros(1, 1, input_dim)
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        return x.view(1, -1).size(1)

    def forward(self, x):
        x = x.unsqueeze(1)

        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x

