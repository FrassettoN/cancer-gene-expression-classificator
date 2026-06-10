import gc
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler


class StackedAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(StackedAutoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, int(input_dim*0.9)),  
            nn.ReLU(),
            nn.Linear(int(input_dim*0.9), int(input_dim*0.75)),  
            nn.ReLU(),
            nn.Linear(int(input_dim*0.75), int(input_dim*0.5))  
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(int(input_dim*0.5), int(input_dim*0.75)),  
            nn.ReLU(),
            nn.Linear(int(input_dim*0.75), int(input_dim*0.9)), 
            nn.ReLU(),
            nn.Linear(int(input_dim*0.9), input_dim),  
            nn.Sigmoid()  
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder(num_epochs, optimizer, model, device, criterion, train_loader):
    model.train()
    
    for epoch in range(num_epochs):
        loss_train = 0.0
        for inputs in train_loader:
            inputs = inputs.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            loss.backward()
            optimizer.step()

            loss_train += loss.item()
            
            del inputs, outputs
            torch.cuda.empty_cache()  

        #print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss_train/len(train_loader)}")


def encoder_output(model, device, data_loader):
    model.eval()
    encoded_features = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)  
            encoded = model.encoder(batch) 
            encoded_features.append(encoded.cpu().numpy())
            
            del batch, encoded
            torch.cuda.empty_cache()

    return np.concatenate(encoded_features, axis=0)

"""
def get_feature_importance(model, method="decoder", top_k=1000):

    with torch.no_grad():
        if method == "decoder":
            weights = model.decoder[0].weight.data.cpu().numpy()  
        elif method == "encoder":
            weights = model.encoder[-1].weight.data.cpu().numpy() 
        else:
            raise ValueError("Metodo non supportato. Usa 'decoder' o 'encoder'.")

        feature_scores = np.sum(np.abs(weights), axis=0)
        important_indices = np.argsort(feature_scores)[::-1][:top_k]
        #important_indices = np.argsort(feature_scores)[-top_k:]

        return important_indices, feature_scores
    
"""

def get_feature_importance(model, method="decoder", threshold=0.1):
    with torch.no_grad():
        if method == "decoder":
            weights = model.decoder[0].weight.data.cpu().numpy()  
        elif method == "encoder":
            weights = model.encoder[-1].weight.data.cpu().numpy() 
        else:
            raise ValueError("Method not supported")

        feature_scores = np.sum(np.abs(weights), axis=0)
        feature_scores = (feature_scores - np.min(feature_scores)) / (np.max(feature_scores) - np.min(feature_scores))
        important_indices = np.where(feature_scores >= threshold)[0]
        
        return important_indices, feature_scores


def filter_features(X_train, X_test, device, batch_size, num_epochs, bottleneck=True):
    
    epsilon = 1e-7
    X_train = np.clip(X_train, 0, 1 - epsilon)
    X_test = np.clip(X_test, 0, 1 - epsilon)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    
    train_loader = torch.utils.data.DataLoader(X_train_tensor, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(X_test_tensor, batch_size=batch_size, shuffle=False)
    
    model = StackedAutoencoder(X_train_tensor.shape[1])
    model = model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

    train_autoencoder(num_epochs, optimizer, model, device, criterion, train_loader)
    
    encoded_train_matrix = encoder_output(model, device, train_loader)
    encoded_test_matrix = encoder_output(model, device, test_loader)

    scaler = MinMaxScaler(feature_range=(0, 1))
    encoded_train_matrix = scaler.fit_transform(encoded_train_matrix)
    encoded_test_matrix = scaler.transform(encoded_test_matrix)
    
    important_indices, _ = get_feature_importance(model, method="decoder", threshold=0.1)

    X_train = X_train[:, important_indices]
    X_test = X_test[:, important_indices]

    del model
    del train_loader
    del test_loader
    del optimizer
    
    gc.collect()
    
    torch.cuda.empty_cache()

    if bottleneck==True:
        return X_train, X_test
    else:
        return encoded_train_matrix, encoded_test_matrix

