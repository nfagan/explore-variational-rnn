import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Encoder network: maps input x (2D) to parameters of q(z|x)
class Encoder(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, latent_dim=2):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)      # Outputs mean of q(z|x)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)  # Outputs log variance of q(z|x)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# Decoder network: maps latent variable z back to reconstruction x_hat (2D)
class Decoder(nn.Module):
    def __init__(self, latent_dim=2, hidden_dim=16, output_dim=2):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        x_hat = self.fc_out(h)
        return x_hat

# Variational Autoencoder (VAE) combining the encoder and decoder
class VAE(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, latent_dim=2):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        # Compute standard deviation
        std = torch.exp(0.5 * logvar)
        # Sample epsilon from a standard normal distribution
        eps = torch.randn_like(std)
        # Return z using the reparameterization trick: z = mu + std * eps
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

# Loss function: reconstruction loss + KL divergence regularizer
def loss_function(x, x_hat, mu, logvar):
    # Mean squared error reconstruction loss (sum over dimensions and batch)
    recon_loss = F.mse_loss(x_hat, x, reduction='sum')
    # KL divergence between q(z|x) and the standard normal prior p(z) ~ N(0, I)
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_divergence

# Example training loop for the VAE on dummy 2D data
def train_vae(model, data_loader, num_epochs=10, learning_rate=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    
    for epoch in range(num_epochs):
        train_loss = 0
        for batch_idx, data in enumerate(data_loader):
            # Assume data is already a tensor of shape (batch_size, 2)
            optimizer.zero_grad()
            x_hat, mu, logvar = model(data)
            loss = loss_function(data, x_hat, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        print(f'Epoch {epoch+1}, Loss: {train_loss/len(data_loader.dataset):.4f}')

# Example usage:
if __name__ == '__main__':
    # Create some dummy 2D data for demonstration
    # For a real application, replace this with your actual dataset
    dummy_data = torch.randn(1000, 2)  # 1000 samples of 2D data
    batch_size = 64
    data_loader = torch.utils.data.DataLoader(dummy_data, batch_size=batch_size, shuffle=True)
    
    # Initialize the VAE model
    vae = VAE(input_dim=2, hidden_dim=16, latent_dim=2)
    
    # Train the VAE
    train_vae(vae, data_loader, num_epochs=20, learning_rate=1e-3)
