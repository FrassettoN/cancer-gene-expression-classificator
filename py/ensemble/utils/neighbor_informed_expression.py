import torch


def neighbor_informed_gene_expression(X_scaled, coords, labels, alpha=1.0, beta=1.0, k=8, sigma=None):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_scaled = X_scaled.to(device) if torch.is_tensor(X_scaled) else torch.tensor(X_scaled, dtype=torch.float, device=device)
    coords   = coords.to(device)   if torch.is_tensor(coords)   else torch.tensor(coords, dtype=torch.float, device=device)
    labels   = labels.to(device)   if torch.is_tensor(labels)   else torch.tensor(labels, dtype=torch.long, device=device)

    n_samples, n_genes = X_scaled.shape

    if k > n_genes - 1:
        raise ValueError(f"k={k} is too large: it must be at most {n_genes-1}")

    # Intra-pixel adjacency
    pixel_coords = coords.round().long().to(device)
    A_intra = torch.zeros((n_genes, n_genes), dtype=torch.float, device=device)
    for p in torch.unique(pixel_coords, dim=0):
        idx = (pixel_coords == p).all(dim=1).nonzero(as_tuple=True)[0]
        for i in idx:
            for j in idx:
                if i != j:
                    A_intra[i, j] = 1.0                 # constant weight

    # Inter-pixel adjacency
    dists = torch.cdist(coords, coords)                 # Inter-pixel adjacency
    dists.fill_diagonal_(float("inf"))                  # Set infinite distance for self-connections
    same_pixel = (pixel_coords.unsqueeze(0) == pixel_coords.unsqueeze(1)).all(-1)
    dists[same_pixel] = float('inf')                    # Ignore intra-pixel connections
    knn_dists, knn_idx = torch.topk(-dists, k, dim=1)   # Now topk selects exactly k nearest neighbors in different pixels
    knn_dists = -knn_dists                              # Revert distances to positive
    A_inter = torch.zeros_like(dists)
    for i in range(n_genes):
        for d, j in zip(knn_dists[i], knn_idx[i]):
            if sigma is None:
                w = 1.0 / (d + 1e-8)
            else:
                w = torch.exp(-(d**2) / (2 * sigma**2))
            A_inter[i, j] = w
    
    # [n_samples, n_genes] + alpha * (X @ A_inter^T) + beta * (X @ A_intra^T)
    new_X = (
        X_scaled
        + alpha * (X_scaled @ A_inter.T)
        + beta * (X_scaled @ A_intra.T)
    )

    return new_X, labels

