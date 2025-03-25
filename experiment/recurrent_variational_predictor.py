import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class RecurrentVariationalPredictor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, latent_dim=2, output_dim=10):
        """
        Args:
            input_dim: Dimensionality of each input xₜ (here 2).
            hidden_dim: Number of hidden units in the LSTMCell.
            latent_dim: Dimensionality of the latent variable zₜ.
            output_dim: Number of classes for the target yₜ.
        """
        super(RecurrentVariationalPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Recurrent cell that processes each time step.
        self.rnn_cell = nn.LSTMCell(input_dim, hidden_dim)
        
        # Inference network: from hidden state, infer latent parameters.
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Prediction network: from [hidden state, zₜ] predict distribution over yₜ.
        self.fc_pred = nn.Linear(hidden_dim + latent_dim, output_dim)
    
    def reparameterize(self, mu, logvar):
        """Applies the reparameterization trick to sample z ~ q(z|x)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x_seq):
        """
        Args:
            x_seq: Tensor of shape [batch_size, seq_len, input_dim]
        Returns:
            logits_seq: Logits for target prediction at each time step,
                        shape [batch_size, seq_len, output_dim]
            mu_seq: Sequence of latent means, shape [batch_size, seq_len, latent_dim]
            logvar_seq: Sequence of latent log variances, same shape as mu_seq.
            z_seq: Sequence of latent variable samples, same shape as mu_seq.
        """
        batch_size, seq_len, _ = x_seq.size()
        
        # Initialize LSTM hidden and cell states.
        hx = torch.zeros(batch_size, self.hidden_dim, device=x_seq.device)
        cx = torch.zeros(batch_size, self.hidden_dim, device=x_seq.device)
        
        logits_seq = []
        mu_seq = []
        logvar_seq = []
        z_seq = []
        
        for t in range(seq_len):
            x_t = x_seq[:, t, :]  # current input xₜ
            hx, cx = self.rnn_cell(x_t, (hx, cx))
            
            # Compute latent distribution parameters from hx.
            mu_t = self.fc_mu(hx)
            logvar_t = self.fc_logvar(hx)
            z_t = self.reparameterize(mu_t, logvar_t)
            
            # Combine hidden state and latent sample for target prediction.
            combined = torch.cat([hx, z_t], dim=1)
            logits_t = self.fc_pred(combined)
            
            # Store outputs for each time step.
            logits_seq.append(logits_t)
            mu_seq.append(mu_t)
            logvar_seq.append(logvar_t)
            z_seq.append(z_t)
        
        # Stack time steps back into tensors.
        logits_seq = torch.stack(logits_seq, dim=1)  # [B, T, output_dim]
        mu_seq = torch.stack(mu_seq, dim=1)            # [B, T, latent_dim]
        logvar_seq = torch.stack(logvar_seq, dim=1)      # [B, T, latent_dim]
        z_seq = torch.stack(z_seq, dim=1)                # [B, T, latent_dim]
        
        return logits_seq, mu_seq, logvar_seq, z_seq

def loss_function(logits_seq, targets, mu_seq, logvar_seq, beta=1.0):
    """
    Computes the overall loss.
    Args:
        logits_seq: [batch_size, seq_len, output_dim] - predicted logits for yₜ.
        targets: [batch_size, seq_len] - true target class indices for each time step.
        mu_seq, logvar_seq: latent distribution parameters at each time step.
        beta: Weighting factor for the KL divergence term.
    Returns:
        total_loss: scalar tensor representing the total loss.
    """
    batch_size, seq_len, output_dim = logits_seq.size()
    
    # Flatten predictions and targets to compute cross entropy.
    logits_flat = logits_seq.view(-1, output_dim)   # [B*T, output_dim]
    targets_flat = targets.view(-1)                   # [B*T]
    ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction='sum')
    
    # KL divergence term per time step.
    kl_div = -0.5 * torch.sum(1 + logvar_seq - mu_seq.pow(2) - logvar_seq.exp())
    
    return ce_loss + beta * kl_div

# Example training loop using dummy sequential data.
def train_model(model, data_loader, num_epochs=10, learning_rate=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for x_seq, y_seq in data_loader:
            # x_seq: [batch_size, seq_len, 2] and y_seq: [batch_size, seq_len] (class indices)
            optimizer.zero_grad()
            logits_seq, mu_seq, logvar_seq, _ = model(x_seq)
            loss = loss_function(logits_seq, y_seq, mu_seq, logvar_seq)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(data_loader.dataset)
        print(f'Epoch {epoch+1}, Loss per sample: {avg_loss:.4f}')

# Example usage:
if __name__ == '__main__':
    # Dummy sequential data: 1000 sequences, each of length 10, with 2D inputs.
    # For targets, we randomly assign one of 10 classes for each time step.
    num_sequences = 1000
    seq_len = 10
    batch_size = 32
    input_dim = 2
    num_classes = 10
    
    x_data = torch.randn(num_sequences, seq_len, input_dim)
    y_data = torch.randint(0, num_classes, (num_sequences, seq_len))
    
    dataset = torch.utils.data.TensorDataset(x_data, y_data)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize the model.
    model = RecurrentVariationalPredictor(input_dim=input_dim, hidden_dim=16, 
                                           latent_dim=2, output_dim=num_classes)
    
    # Train the model.
    train_model(model, data_loader, num_epochs=20, learning_rate=1e-3)
