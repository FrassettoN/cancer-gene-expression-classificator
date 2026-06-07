import torch.nn.functional as F
from torch import nn


class CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):

        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.fc1 = None
        self.fc2 = None
        
        self.num_classes = num_classes
        
        self.dropout = nn.Dropout(0.5)
        
        self.initialized = False
        
    def _initialize_fc_layers(self, x):

        fc_input_size = x.shape[1]
        
        self.fc1 = nn.Linear(fc_input_size, 512).to(x.device)
        self.fc2 = nn.Linear(512, self.num_classes).to(x.device)
        
        self.initialized = True
        
    def _adaptive_max_pool(self, x, kernel_size=2):

        if x.size(2) < kernel_size or x.size(3) < kernel_size:
            return x
        return F.max_pool2d(x, kernel_size)
    
    def forward(self, x):

        original_shape = x.shape
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self._adaptive_max_pool(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self._adaptive_max_pool(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self._adaptive_max_pool(x, 2)
        
        x = x.view(x.size(0), -1)
        
        if not self.initialized:
            self._initialize_fc_layers(x)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

