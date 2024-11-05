"""
VAE-based counterfactual data augmentation module from CLAIRE.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim + 1, hidden_dim)  # Input: {X, Y}
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)  # Latent mean
        self.fc_var = nn.Linear(hidden_dim, latent_dim)  # Latent log variance
        self.activation = nn.LeakyReLU()
        
    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        h = self.activation(self.fc1(xy))
        mu = self.fc_mu(h)
        logvar = self.fc_var(h)
        return mu, logvar
    
    
class Decoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + 1, hidden_dim)  # Input: {H, S}
        self.fc2 = nn.Linear(hidden_dim, output_dim + 1)  # Output: {X, Y}
        self.activation = nn.LeakyReLU()
    
    def forward(self, h, s):
        hs = torch.cat([h, s], dim=1)
        z = self.activation(self.fc1(hs))
        xy_recon = self.fc2(z)
        x_recon = xy_recon[:, :-1]
        y_recon = xy_recon[:, -1].unsqueeze(1)  
        return x_recon, y_recon


class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim, hidden_dim)
        self.decoder = Decoder(latent_dim, input_dim, hidden_dim)

    def forward(self, x, y, s):
        mu, logvar = self.encoder(x, y)
        h = self.reparameterize(mu, logvar)  # sample from the latent space in a differentiable way
        x_recon, y_recon = self.decoder(h, s)
        return x_recon, y_recon, mu, logvar, h
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, sigma^2) using mu + sigma * epsilon, epsilon ~ N(0,I) at train time. 
            This is crucial for training the VAE because it allows backpropagation through the stochastic latent variable.
        During inference, we don't need to sample from the latent space; directly use the mean of the latent distribution.
            This ensures that the model is deterministic during inference, which is important for generating counterfactuals.
        """
        if self.training:  # Training mode: use reparameterization trick
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:  # Inference mode: use mean directly for deterministic outputs
            return mu

    
def vae_loss(x_true, y_true, x_recon, y_recon, mu, logvar, kld_weight=1.0):
    """
    Compute the total VAE loss, which consists of two components:
        1. Reconstruction loss: Measures how well the decoder reconstructs {X,Y} from the 
            latent embeddings H and sensitive attribute S.
        2. KL divergence: Encourages the learned latent space distribution to be close to 
            a standard normal distribution N(0,I).
    
    Args:
        x_true, y_true: Original input data {X,Y}
        x_recon, y_recon: Reconstructed data from decoder
        mu: Mean of latent distribution from encoder
        logvar: Log variance of latent distribution from encoder
    
    Returns:
        Total VAE loss combining reconstruction and KL divergence.
    """
    # Reconstruction loss for continuous features X (MSE), averaged over batch 
    recon_loss_x = F.mse_loss(x_recon, x_true, reduction='mean')
    # Reconstruction loss for binary target Y (Sigmoid + Binary Cross-Entropy), averaged over batch 
    recon_loss_y = F.binary_cross_entropy_with_logits(y_recon, y_true, reduction='mean')
    # Total reconstruction loss, averaged over batch 
    recon_loss = recon_loss_x + recon_loss_y
    
    # KL divergence loss, summed over latent dimensions and averaged over batch
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    
    return recon_loss + kld_weight * kld_loss


def mmd_loss(x, y, kernel='rbf', bandwidth=1.0):
    """
    Compute the Maximum Mean Discrepancy (MMD) between two distributions.
    This will work even if the number of samples in each group is different.
    """
    # Compute pairwise squared Euclidean distances between all points in x and y
    xx = torch.mm(x, x.t())  # shape: (n1, n1)
    yy = torch.mm(y, y.t())  # shape: (n2, n2)
    xy = torch.mm(x, y.t())  # shape: (n1, n2)
    
    # Compute squared norms for each row in x and y
    rx = xx.diag().unsqueeze(0)  # shape: (1, n1)
    ry = yy.diag().unsqueeze(0)  # shape: (1, n2)
    
    if kernel == 'rbf':
        # Compute RBF kernel for all pairs
        Kxx = torch.exp(- (rx.t() + rx - 2 * xx) / (2 * bandwidth))  # shape: (n1, n1)
        Kyy = torch.exp(- (ry.t() + ry - 2 * yy) / (2 * bandwidth))  # shape: (n2, n2)
        Kxy = torch.exp(- (rx.t() + ry - 2 * xy) / (2 * bandwidth))  # shape: (n1, n2)
    else:
        raise ValueError('Unsupported kernel type')
    
    # Return MMD loss
    return Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()


def claire_m_loss(vae_loss, embeddings_by_group, alpha=2.0):
    """
    Compute CLAIRE loss with MMD penalty for distribution matching (CLAIRE-M).
    
    Args:
        vae_loss: The VAE loss (reconstruction loss + KL divergence).
        embeddings_by_group: A dictionary mapping sensitive groups to their latent embeddings.
        alpha: Weighting factor for the MMD penalty, controls the importance of the distribution matching term.
            The default value is 2.0 from the original CLAIRE paper.
    
    Returns:
        Total CLAIRE-M loss combining VAE loss and MMD penalty.
    """
    mmd_penalty = 0.0
    sensitive_groups = list(embeddings_by_group.keys())
    
    num_pairs = len(sensitive_groups) * (len(sensitive_groups) - 1) / 2
    
    for i in range(len(sensitive_groups)):
        for j in range(i + 1, len(sensitive_groups)):
            s_i = sensitive_groups[i]
            s_j = sensitive_groups[j]
            mmd_penalty += mmd_loss(embeddings_by_group[s_i], embeddings_by_group[s_j])
    
    mmd_penalty /= num_pairs  # Average MMD over all pairs of sensitive groups
    
    alpha = max(0.0, alpha)  # Ensure alpha is non-negative
    return vae_loss + alpha * mmd_penalty


def train(model: VAE, train_dataloader: DataLoader, num_epochs: int = 100, learning_rate: float = 1e-4, 
          kld_weight: float = 1.0, alpha: float = 2.0, device=DEVICE) -> VAE:
    """
    Train the VAE model using the given training data and hyperparameters.
    The Adam optimizer is used for training.

    Args:
        model: The VAE model to train.
        train_dataloader: DataLoader for the training dataset.
        num_epochs: Defaults to 100.
        kld_weight: Defaults to 1.0.
        alpha: Defaults to 2.0.
        device: Defaults to DEVICE.

    Returns:
        The trained VAE model.
    """
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()  # Switch to training mode
    step = 0  # Global step count over all epochs
    epoch_losses = []
    
    for epoch in trange(num_epochs, unit="epochs"):
        total_loss = 0.0
        
        with tqdm(train_dataloader, unit="batches") as itr:
            for step, (X_batch, Y_batch, S_batch) in enumerate(itr, start=step + 1):
                X_batch, Y_batch, S_batch = \
                    X_batch.to(device).float(), Y_batch.to(device).float(), S_batch.to(device).float()
                
                # Zero out the gradients before passing in a new batch
                optimizer.zero_grad()
                
                # Forward pass (get reconstructed input and latent embeddings)
                X_recon, Y_recon, mu, logvar, H_batch = \
                    model.forward(X_batch, Y_batch, S_batch)
                
                # Compute VAE loss
                loss_vae = vae_loss(X_batch, Y_batch, X_recon, Y_recon, mu, logvar, kld_weight)
                
                sensitive_groups = torch.unique(S_batch)
                embeddings_by_group = {}
                for s in sensitive_groups:
                    mask_s = (S_batch == s).nonzero(as_tuple=True)[0]
                    embeddings_by_group[s.item()] = H_batch[mask_s]
                
                # Compute CLAIRE-M loss
                loss = claire_m_loss(loss_vae, embeddings_by_group, alpha)
                
                # Backpropagation and optimization step
                loss.backward()  # Compute gradients
                optimizer.step()  # Update model parameters
                
                total_loss += loss.item()
            
        avg_epoch_loss = total_loss / len(train_dataloader)
        epoch_losses.append(avg_epoch_loss)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}')
    
    model.eval()  # Switch to inference mode 
    return model, epoch_losses


class CounterfactualDataGenerator:
    def __init__(self, vae: VAE, sensitive_groups: list[str | int], K: int = 10) -> None:
        """Initialize the counterfactual data generator.

        Args:
            vae: The trained VAE model with an encoder that encodes {X, Y} into latent space H 
                and a decoder that decodes {H, S} back into {X, Y}.
            sensitive_groups: Unique values of the sensitive attribute.
            K: Number of samples to generate from the latent space for each instance. Defaults to 10.
        """
        self.vae = vae  # Trained VAE model
        self.vae.eval()  # Make sure it is in inference mode
        self.sensitive_groups = sensitive_groups  # Unique values of the sensitive attribute
        self.K = K  # Number of samples to generate from the latent space for each instance
    
    def generate_counterfactuals(self, X: torch.Tensor, Y: torch.Tensor) -> dict[str | int, np.ndarray]:
        """
        Generate counterfactuals for a batch of {X, Y} instances.
        
        Args:
            X: Input features (batch of X values).
            Y: Input labels (batch of Y values).
        
        Returns:
            Two dictionaries (X_CF_s, Y_CF_s) containing counterfactual versions of X and Y 
            for each sensitive group s, respectively.
        """
        # Encode {X, Y} into latent space H using the encoder
        mu, logvar = self.vae.encoder(X.float(), Y.float())
        
        # Generate K samples from the latent space using reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn(self.K, *mu.size()).to(mu.device)  # shape: (K, batch_size, latent_dim)
        H_samples = mu.unsqueeze(0) + eps * std.unsqueeze(0)  # shape: (K, batch_size, latent_dim)
        
        # Initialize dictionaries to store counterfactuals for each sensitive group
        X_CF_s = {}
        Y_CF_s = {}
        
        for s in self.sensitive_groups:  # for each value of the sensitive attribute  
            S = torch.full(size=(X.shape[0], 1), fill_value=s)  # shape: (batch_size, 1)
            
            # Initialize lists to store decoded outputs for K samples
            X_decoded = []
            Y_decoded = []
            
            # Decode each sample H_k with sensitive attribute s
            for k in range(self.K):
                H_k = H_samples[k]  # shape: (batch_size, latent_dim)
                
                # Decode {H_k, s} -> {X_k^CF, Y_k^CF}
                X_recon, Y_recon = self.vae.decoder(H_k.float(), S.float())
                
                X_decoded.append(X_recon)
                Y_decoded.append((torch.sigmoid(Y_recon) > 0.5).float())  # Binarize Y
            
            # Aggregate (mean) over K samples to get counterfactuals
            X_CF_s[s] = torch.mean(torch.stack(X_decoded), dim=0).cpu().detach().numpy()
            Y_CF_s[s] = (torch.mean(torch.stack(Y_decoded), dim=0) > 0.5).float().cpu().detach().numpy()

        return X_CF_s, Y_CF_s


def to_tensor(input, device=DEVICE):
    """Convert an array-like input to a PyTorch tensor."""
    if isinstance(input, torch.Tensor):
        return torch.to(device)
    
    return torch.tensor(input, dtype=torch.float32).to(device)
