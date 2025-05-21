import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class RecurrentVAE(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, latent_dim=2):
        super(RecurrentVAE, self).__init__()
        self.hidden_dim = hidden_dim

        # Recurrent encoder: LSTMCell processes one time step at a time.
        self.rnn_cell = nn.LSTMCell(input_dim, hidden_dim)
        
        # From hidden state, infer latent distribution parameters.
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder: reconstructs the input using both hidden state and latent variable.
        self.fc_dec = nn.Linear(hidden_dim + latent_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # standard deviation
        eps = torch.randn_like(std)    # sample epsilon from standard normal
        return mu + eps * std

    def forward(self, x_seq):
        """
        x_seq: Tensor of shape [batch_size, seq_len, input_dim]
        Returns:
          x_recon_seq: Reconstructed sequence, same shape as x_seq.
          mu_seq: Means at each time step, shape [batch_size, seq_len, latent_dim].
          logvar_seq: Log variances at each time step, shape [batch_size, seq_len, latent_dim].
          z_seq: Sampled latent variables for each time step.
        """
        batch_size, seq_len, _ = x_seq.size()
        
        # Initialize hidden and cell states for LSTMCell.
        hx = torch.zeros(batch_size, self.hidden_dim, device=x_seq.device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=x_seq.device)
        
        x_recon_seq = []
        mu_seq = []
        logvar_seq = []
        z_seq = []

        # Process each time step sequentially.
        for t in range(seq_len):
            x_t = x_seq[:, t, :]  # Current input of shape [batch_size, input_dim]
            
            # Update recurrent state.
            hx, cx = self.rnn_cell(x_t, (hx, cx))
            
            # Compute latent parameters from the hidden state.
            mu_t = self.fc_mu(hx)
            logvar_t = self.fc_logvar(hx)
            z_t = self.reparameterize(mu_t, logvar_t)
            
            # Decode: combine hidden state and latent variable.
            dec_input = torch.cat([hx, z_t], dim=1)
            x_recon_t = self.fc_dec(dec_input)
            
            # Store outputs.
            x_recon_seq.append(x_recon_t)
            mu_seq.append(mu_t)
            logvar_seq.append(logvar_t)
            z_seq.append(z_t)
        
        # Stack lists into tensors along the time dimension.
        x_recon_seq = torch.stack(x_recon_seq, dim=1)  # [batch_size, seq_len, input_dim]
        mu_seq = torch.stack(mu_seq, dim=1)            # [batch_size, seq_len, latent_dim]
        logvar_seq = torch.stack(logvar_seq, dim=1)      # [batch_size, seq_len, latent_dim]
        z_seq = torch.stack(z_seq, dim=1)                # [batch_size, seq_len, latent_dim]
        
        return x_recon_seq, mu_seq, logvar_seq, z_seq

def loss_function(x_seq, x_recon_seq, mu_seq, logvar_seq):
    """
    Computes the loss for a sequence.
    The reconstruction loss uses mean squared error (MSE) over all time steps.
    The KL divergence term is computed for each time step and summed.
    """
    # Flatten the sequences: combine batch and time dimensions.
    recon_loss = F.mse_loss(x_recon_seq.view(-1, x_recon_seq.size(-1)), 
                            x_seq.view(-1, x_seq.size(-1)),
                            reduction='sum')
    
    # Compute KL divergence per time step.
    # For each latent dimension: 0.5 * sum( sigma^2 + mu^2 - 1 - log(sigma^2) )
    kl_div = -0.5 * torch.sum(1 + logvar_seq - mu_seq.pow(2) - logvar_seq.exp())
    
    return recon_loss + kl_div

# Example training loop using dummy sequential 2D data.
def train_rvae(model, data_loader, num_epochs=10, learning_rate=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, x_seq in enumerate(data_loader):
            # x_seq should have shape [batch_size, seq_len, 2]
            optimizer.zero_grad()
            x_recon_seq, mu_seq, logvar_seq, _ = model(x_seq)
            loss = loss_function(x_seq, x_recon_seq, mu_seq, logvar_seq)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(data_loader.dataset)
        print(f'Epoch {epoch+1}, Loss per sample: {avg_loss:.4f}')

# Example usage:
if __name__ == '__main__':
    # Create dummy sequential 2D data: 1000 sequences, each of length 10.
    dummy_data = torch.randn(1000, 10, 2)
    batch_size = 32
    data_loader = torch.utils.data.DataLoader(dummy_data, batch_size=batch_size, shuffle=True)
    
    # Initialize the Recurrent VAE.
    model = RecurrentVAE(input_dim=2, hidden_dim=16, latent_dim=2)
    
    # Train the model.
    train_rvae(model, data_loader, num_epochs=20, learning_rate=1e-3)
