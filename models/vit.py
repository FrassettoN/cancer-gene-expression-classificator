"""
Source:
1. C. Bircanoglu, "Step-by-step Implementation of Vision Transformers in PyTorch,"
   Medium, Online: https://cenk-bircanoglu.medium.com/step-by-step-implementation-of-vision-transformers-in-pytorch-5aaf0ccbc8d3
"""

import math
import torch
import torch.nn.functional as F
from torch import nn


# Self Attention Module
class SelfAttention(nn.Module):
    def __init__(self, embedding_dim, qkv_dim):
        super(SelfAttention, self).__init__()

        self.embedding_dim = embedding_dim   # Embedding dimension
        self.qkv_dim = qkv_dim               # Dimension of key, query, value

        self.W = nn.Parameter(torch.empty(1, embedding_dim, int(3 * qkv_dim)).normal_(std=0.02))

    def forward(self, embeddings):
        # Calculate query, key and value projection
        qkv = torch.matmul(embeddings, self.W)
        q = qkv[:, :, :self.qkv_dim]
        k = qkv[:, :, self.qkv_dim:self.qkv_dim*2 ]
        v = qkv[:, :, self.qkv_dim*2:]
        
        # Calculate attention weights by applying a softmax to the dot product of all queries with all keys
        attention_weights = F.softmax(torch.matmul(q, torch.transpose(k, -2, -1) ) / math.sqrt(self.qkv_dim), dim=1)
        
        # Calculate attention values and return
        return torch.matmul(attention_weights, v)
    

# Multi-Head Attention Module
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads            
        self.embedding_dim = embedding_dim          # Embedding dimension

        self.qkv_dim = embedding_dim // num_heads   # Dimension of key, query, and value can be calculated with embedding_dim and num_of_heads

        # Initialise self-attention modules num_heads times
        self.multi_head_attention = nn.ModuleList([SelfAttention(embedding_dim, self.qkv_dim) for _ in range(num_heads)])

        # Initialise weight matrix
        self.W = nn.Parameter(torch.empty(1, num_heads * self.qkv_dim, embedding_dim).normal_(std=0.02))

    def forward(self, x):
        # Self-attention scores for each head
        attention_scores = [attention(x) for attention in self.multi_head_attention]

        # The outputs from all attention heads are concatenated and linearly transformed
        Z = torch.cat(attention_scores, -1)

        # This step ensures that the model can consider a comprehensive set of relationships captured by different heads
        return torch.matmul(Z, self.W)


class MLP(nn.Module):
    def __init__(self, embedding_dim, hidden_dim):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
                            nn.Linear(embedding_dim, hidden_dim),
                            nn.GELU(),
                            nn.Linear(hidden_dim, embedding_dim)
                   )

    def forward(self, x):
        return self.mlp(x)


# Transformer Encoder Module
class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, num_heads, hidden_dim, dropout):
        super(TransformerEncoder, self).__init__()

        self.multi_head_attention = MultiHeadAttention(embedding_dim, num_heads)
        self.mlp = MLP(embedding_dim, hidden_dim)

        self.layer_norm1 = nn.LayerNorm(embedding_dim)
        self.layer_norm2 = nn.LayerNorm(embedding_dim)

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

    def forward(self, embeddings):
        # Applying dropout
        dropout_embeddings = self.dropout1(embeddings)

        # Layer normalization
        normalized_embeddings = self.layer_norm1(dropout_embeddings)

        # Calculation of multi-head attention
        attention_scores = self.multi_head_attention(normalized_embeddings)

        # Applying the second dropout
        dropout_attention_scores = self.dropout2(attention_scores)

        # Residual connection with second dropout output and initial input
        residuals_embeddings = embeddings + dropout_attention_scores

        # Apply layer normalization
        normalized_residuals_embeddings = self.layer_norm2(residuals_embeddings)

        # Aply MLP 
        transformed_results = self.mlp(normalized_residuals_embeddings)

        # Applying the third dropout
        dropout_transformed_results = self.dropout3(transformed_results)

        # Residual connection with last dropout output and first residual output
        output = residuals_embeddings + dropout_transformed_results

        return output


class MLPHead(nn.Module):
    def __init__(self, embedding_dim, num_classes, is_train=True):
        super(MLPHead, self).__init__()
        self.num_classes = num_classes

        # This part is taken from torchvision implementation
        if is_train:
            self.head = nn.Sequential(
                                    nn.Linear(embedding_dim, 3072),   # Hidden layer
                                    nn.Tanh(),
                                    nn.Linear(3072, num_classes)      # Output layer
                            )
        else:
            # Single linear layer
            self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        return self.head(x)


# VisionTranformer Module
class VisionTransformer(nn.Module):
    def __init__(self, patch_size=16, image_size=224, C=3,
                     num_layers=12, embedding_dim=768, num_heads=12, hidden_dim=3072,
                            dropout_prob=0.1, num_classes=10):
        super(VisionTransformer, self).__init__()

        self.patch_size = patch_size
        self.C = C

        # Get the number of patches of the image
        self.num_patches = int(image_size ** 2 / patch_size ** 2)   # (width * height) / (patch_size**2)

        # Trainable linear projection for mapping dimension of patches (weight matrix E)
        self.W = nn.Parameter(torch.empty(1, patch_size * patch_size * C, embedding_dim).normal_(std=0.02))

        # Position embeddings
        self.positional_embeddings = nn.Parameter(torch.empty(1, self.num_patches + 1, embedding_dim).normal_(std=0.02))

        # Learnable class tokens
        self.class_tokens = nn.Parameter(torch.rand(1, embedding_dim))

        # Transformer encoder
        self.transformer_encoder = nn.Sequential(*[
            TransformerEncoder(embedding_dim, num_heads, hidden_dim, dropout_prob) for _ in range(num_layers)
        ])

        # MLP head
        #self.mlp_head = MLPHead(embedding_dim, num_classes)
        self.mlp_head_1 = MLPHead(embedding_dim, 1048)
        self.mlp_head_2 = MLPHead(1048, num_classes)

    def forward(self, images):
        # Get patch size and channel size
        P, C = self.patch_size, self.C

        # Create image patches
        patches = images.unfold(1, C, C).unfold(2, P, P).unfold(3, P, P).contiguous().view(images.size(0), -1, C * P * P).float()

        # Patch embeddings
        patch_embeddings = torch.matmul(patches , self.W)

        # Class token + patch_embeddings
        batch_class_token = self.class_tokens.expand(patch_embeddings.shape[0], -1, -1)
        patch_embeddings_with_class_token = torch.cat([batch_class_token, patch_embeddings], dim=1)

        # Add positional embedding
        embeddings = patch_embeddings_with_class_token + self.positional_embeddings

        # Execute Transformer encoders
        transformer_encoder_output = self.transformer_encoder(embeddings)

        # Classifier "token" as used by standard language architectures
        output_class_token = transformer_encoder_output[:, 0]

        intermediate_output = self.mlp_head_1(output_class_token)

        return self.mlp_head_2(intermediate_output)

