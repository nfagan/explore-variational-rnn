import torch
import torch.nn as nn
import torch.optim as optim

# Custom RNN cell that repeats its state update.
class RepeatRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, repetitions=2):
        super().__init__()
        self.repetitions = repetitions
        self.rnn_cell = nn.RNNCell(input_size, hidden_size)

    def forward(self, input, hx):
        for _ in range(self.repetitions):
            hx = self.rnn_cell(input, hx)
        return hx

# RepeatRNN model: processes the repeated input sequence.
class RepeatRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, repetitions=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = RepeatRNNCell(input_size, hidden_size, repetitions)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, inputs):
        # inputs shape: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = inputs.size()
        hx = torch.zeros(batch_size, self.hidden_size, device=inputs.device)
        for t in range(seq_len):
            hx = self.cell(inputs[:, t, :], hx)
        out = self.fc(hx)  # shape: (batch_size, 1)
        return out

# Data generation for the parity task.
# Each sample is a binary vector of length 64.
# The parity (0 for even, 1 for odd) is computed over the vector.
# The vector is repeated num_reps times to create a sequence.
def generate_parity_data(batch_size, vector_length=64, num_reps=2):
    # Generate random binary vectors of shape (batch_size, vector_length)
    x = torch.randint(0, 2, (batch_size, vector_length)).float()
    # Compute parity: 0 if even, 1 if odd.
    # We convert targets to float and unsqueeze to match output shape (batch_size, 1)
    target = (x.sum(dim=1) % 2).float().unsqueeze(1)
    # Repeat each vector along a new sequence dimension.
    # Resulting shape: (batch_size, num_reps, vector_length)
    x_seq = x.unsqueeze(1).repeat(1, num_reps, 1)
    return x_seq * 2 - 1, target

# Hyperparameters
input_size    = 64    # Length of each input vector.
hidden_size   = 128   # Hidden state dimension.
output_size   = 1     # Single logit output for binary classification.
repetitions   = 8     # Number of state updates per time step.
num_reps      = 1     # Number of times the input is repeated (sequence length).
batch_size    = 128
num_epochs    = 1000000
learning_rate = 0.001

# Initialize model, loss function, and optimizer.
model = RepeatRNN(input_size, hidden_size, output_size, repetitions)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop.
for epoch in range(num_epochs):
    model.train()
    data, targets = generate_parity_data(batch_size, input_size, num_reps)
    optimizer.zero_grad()
    outputs = model(data)  # outputs shape: (batch_size, 1)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        model.eval()
        with torch.no_grad():
            data_val, targets_val = generate_parity_data(batch_size, input_size, num_reps)
            outputs_val = model(data_val)
            # Apply sigmoid to obtain probabilities, then threshold at 0.5.
            predictions = (torch.sigmoid(outputs_val) >= 0.5).float()
            accuracy = (predictions == targets_val).float().mean().item()
            print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}, Accuracy = {accuracy*100:.2f}%")
