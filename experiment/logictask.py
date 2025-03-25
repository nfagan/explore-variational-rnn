import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

LOGIC_OPERATIONS = {
    'AND': lambda x, y: x & y,  
    # Logical AND: True if both x and y are True; otherwise, False.
    
    'OR': lambda x, y: x | y,  
    # Logical OR: True if at least one of x or y is True.
    
    'XOR': lambda x, y: x ^ y,  
    # Exclusive OR: True if x and y are different, False if they are the same.
    
    'NAND': lambda x, y: ~(x & y) & 1,  
    # NOT AND: Inverse of AND; True unless both x and y are True.
    
    'NOR': lambda x, y: ~(x | y) & 1,  
    # NOT OR: Inverse of OR; True only if both x and y are False.
    
    'XNOR': lambda x, y: ~(x ^ y) & 1,  
    # Logical Equivalence (XNOR): True if x and y are the same.
    
    'IMPLIES': lambda x, y: (~x | y) & 1,  
    # Logical Implication (if/then, P → Q): False only when x is True and y is False.
    
    'REVERSE_IMPLIES': lambda x, y: (x | ~y) & 1,  
    # Reverse Implication (then/if, Q → P): False only when y is True and x is False.
    
    'XQ': lambda x, y: (~x & y) & 1,  
    # Custom logic from the paper: True only if x is False and y is True.
    
    'ABJ': lambda x, y: (x & ~y) & 1  
    # Material Nonimplication (Abjunction, P ⊅ Q): True if x is True and y is False.
}

class LogicDataset(torch.utils.data.Dataset):
    """
    A logic operation dataset.
    """

    def __init__(
            self,
            num_samples = 10000,
            num_operations = 10,
            seq_len = 10,
            match_non_sequence_tasks = False,
            per_position_targets = False
        ):
        """
        Initialize the dataset.
        """

        self.num_samples = num_samples
        self.num_operations = num_operations
        self.seq_len = seq_len
        self.operation_list = list(LOGIC_OPERATIONS.keys())
        self.match_non_sequence_tasks = match_non_sequence_tasks
        self.per_position_targets = per_position_targets
        
        # generate data
        self.generate_data()


    def generate_data(self):
        """
        Generate dataset.
        """
        
        self.data = []
        self.labels = []
        self.symbols = []
        self.operations = []
        
        for _ in range(self.num_samples):
            # sample initial variables
            s1, s2 = np.random.choice([0, 1], size = 2)

            # sample operations
            operation_indices = np.random.randint(0, len(self.operation_list), size = self.num_operations)

            # compute ground truth output
            result = s1
            for idx in operation_indices:
                result = LOGIC_OPERATIONS[self.operation_list[idx]](result, s2)
                result = int(result)

                if result not in [0, 1]:
                    raise ValueError('Result is not boolean.')

            # convert operations to one-hot encoding using torch
            operations = F.one_hot(torch.tensor(operation_indices), num_classes = len(self.operation_list)).float() # (seq_len, num_operations)
            operations = operations.reshape(-1) # (seq_len * num_operation,)
            
            # create tensors
            input = torch.cat([torch.tensor([s1, s2], dtype = torch.float32), operations], dim = 0) # (seq_len * num_operations + 2,)
            label = torch.tensor(result, dtype = torch.long) # integer class index

            # append data and label
            self.data.append(input)
            self.labels.append(label)
            self.symbols.append([s1, s2])
            self.operations.append(list(operation_indices))
        
        self.data = torch.stack(self.data)
        self.data = self.data.unsqueeze(1).repeat(1, self.seq_len, 1) # (num_samples, seq_len, feature_size)
        self.labels = torch.tensor(self.labels) # (num_samples)

        self.symbols = torch.tensor(self.symbols)
        self.operations = torch.tensor(self.operations)


    def __len__(self):
        return self.num_samples


    def __getitem__(self, idx):
        if self.match_non_sequence_tasks:
            x = self.data[idx]
            y = self.labels[idx]
            y = y.unsqueeze(0).repeat(x.shape[0]).unsqueeze(1)
            m = torch.ones_like(y)
            return x, y, m
        return self.data[idx], self.labels[idx]

class Trainer:
    """
    A trainer class.
    """
    
    def __init__(
            self,
            model,
            train_loader,
            lr = 1e-3,
            device = 'cpu'
        ):
        """
        Initialize the trainer.
        """

        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr = lr)


    def train(self, num_epochs):
        """
        Train the network.
        """
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs, _ = self.model(inputs) # (batch_size, seq_len, 2)
                loss = self.criterion(outputs[:, -1, :], targets)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            print(f'Epoch {epoch}, Loss: {epoch_loss / len(self.train_loader):.4f}')
    
class LogicGRU(nn.Module):
    """
    A GRU network class.
    """

    def __init__(
            self,
            input_size,
            hidden_size = 128,
        ):
        """
        Initialize the network.
        """
        
        super(LogicGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, batch_first = True)
        self.fc = nn.Linear(hidden_size, 2)
    

    def forward(self, x):
        """
        Forward the network.
        
        Args:
            x: a torch.tensor with a shape of (batch_size, seq_len, input_size).
        
        Returns:
            outputs: a torch.tensor with a shape of (batch_size, seq_len, 2)
            hiddens: a torch.tensor with a shape of (batch_size, seq_len, hidden_size)
        """

        batch_size = x.size(0)
        hidden_init = torch.zeros(1, batch_size, self.hidden_size) # (layer_size, batch_size, hidden_size)
        hiddens, _ = self.gru(x, hidden_init) # (batch_size, seq_len, hidden_size)
        outputs = self.fc(hiddens) # (batch_size, seq_len, 2)

        return outputs, hiddens
    
if __name__ == '__main__':
  # initialize dataset and dataloader
  dataset = LogicDataset(
      num_samples = 10000,
      num_operations = 10,
      seq_len = 10,
  )
  train_loader = torch.utils.data.DataLoader(dataset, batch_size = 1024, shuffle = True)

  # define model and trainer
  model = LogicGRU(
      input_size = 102,
      hidden_size = 128,
  )
  trainer = Trainer(
      model = model,
      train_loader = train_loader,
      lr = 1e-3,
  )

  # train the model
  trainer.train(num_epochs = 50)

  eval_dataset = LogicDataset(
      num_samples = 10000,
      num_operations = 10,
      seq_len = 10,
  )
  eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size = 1024, shuffle = True)

  with torch.no_grad():
      inputs = eval_loader.dataset.data
      targets = eval_loader.dataset.labels

      outputs, hiddens = model(inputs) # (num_samples, seq_len, 2) / (num_samples, seq_len, hidden_size)

      probs = torch.softmax(outputs, dim = -1)
      idxs = torch.argmax(probs, dim = -1)

      t = idxs[:, -1]
      acc = (targets == t).sum() / t.shape[0]
      print(f'Acc: {acc}')