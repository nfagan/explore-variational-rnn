import random
import torch
import torch.nn as nn
from typing import Tuple

"""
Parity task
"""

def generate_parity_task_sequence(n: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  np = torch.randint(1, n + 1, (1,))
  nn = torch.min(torch.randint(1, n + 1, (1,)), n - np)
  # nz = n - np - nn
  v = torch.zeros((n,))
  v[:np] = 1.
  v[np:np+nn] = -1.
  v[np+nn:] = 0.
  v[:np+nn] = v[torch.randperm(np+nn)]
  # v = v[torch.randperm(n)]
  """
  "The corresponding target was 1 if there was an odd number of ones and 0 if there was an
  even number of ones"
  """
  t = ((np % 2) == 0).type(torch.float32)
  mask = torch.ones((1,))
  return v, t, mask

"""
Addition task
"""

def generate_addition_task_sequence(
  seq_len: int, max_num_digits: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """
  """
  x = torch.zeros((max_num_digits * 10, seq_len))
  y = torch.zeros((max_num_digits + 1, seq_len), dtype=torch.long)
  # y = torch.zeros((max_num_digits * 11, seq_len)) # +1 for "end of number"

  seq_mask = torch.ones((seq_len, 1))
  seq_mask[0] = 0

  for i in range(seq_len):
    num_digits = 1 + torch.randint(0, max_num_digits, (1,))
    for d in range(num_digits):
      digit = torch.randint(0, 10, (1,))
      x[i*10:(i+1)*10, i] = nn.functional.one_hot(digit, 10)
      s = digit if i == 0 else s + digit
      ss = str(s.item())
      r = len(ss)
      for k in range(r):
        y[k, i] = int(ss[k])
        # y[k*11:(k+1)*11, i] = nn.functional.one_hot(torch.tensor(int(ss[k])), 11)
      for k in range((max_num_digits + 1) - r):
        y[k+r, i] = 10
        # y[(k+r)*11:(k+r+1)*11, i] = nn.functional.one_hot(torch.tensor(10), 11)
  return x.T, y.T, seq_mask

"""
Logic task
"""

# Define the list of operations in fixed order.
ops_list = ['NOR', 'Xq', 'ABJ', 'XOR', 'NAND', 'AND', 'XNOR', 'if/then', 'then/if', 'OR']

# Define the operations as functions mapping (x,y) in {0,1} x {0,1} -> {0,1}
ops = {
  'NOR':    lambda x, y: int((not x) and (not y)),
  'Xq':     lambda x, y: 1 if (x == 1 and y == 0) else 0,
  'ABJ':    lambda x, y: 1 if (x == 0 and y == 1) else 0,
  'XOR':    lambda x, y: int(x != y),
  'NAND':   lambda x, y: 1 if not (x and y) else 0,
  'AND':    lambda x, y: int(x and y),
  'XNOR':   lambda x, y: 1 if x == y else 0,
  'if/then':lambda x, y: int((not x) or y),
  'then/if':lambda x, y: int((not y) or x),
  'OR':     lambda x, y: int(x or y)
}

def generate_logic_task_examplar(b0, num_ops):
  """
  """
  # 1. Sample the two initial binary numbers.
  b1 = random.randint(0, 1)
  
  # 2. Decide how many operations B to apply (from 1 to 10).
  B = random.randint(1, num_ops)
  
  # 3. Prepare the 10 chunks of size 10.
  # For the first B chunks, set a one-hot encoding for a randomly chosen operation (index 0 to 9).
  chunks = []
  for i in range(num_ops):
    if i < B:
      op_index = random.randint(0, 9)
      one_hot = [0] * 10
      one_hot[op_index] = 1
      chunks.extend(one_hot)
    else:
      # For remaining chunks, append 10 zeros.
      chunks.extend([0] * 10)
  
  # 4. Assemble the full input vector (first two elements are b1 and b0, then the 10 chunks).
  input_vec = [float(b1), float(b0)] + [float(x) for x in chunks]  # total length 2 + 100 = 102

  # 5. Compute the target by recursively applying the operations.
  b_prev = b0
  b_curr = b1

  # accumulator = b1
  # For each of the first B chunks, decode the one-hot into an operation index.
  # We iterate over chunks in order.
  for i in range(B):
    # Each chunk is 10 numbers; extract the chunk.
    start = 2 + i * 10  # after the first two elements
    chunk = input_vec[start:start+10]
    # Get the index of the 1 in the one-hot vector.
    op_index = chunk.index(1)  # assumes exactly one 1 per chunk
    op_name = ops_list[op_index]
    # Update the accumulator.
    res = ops[op_name](b_curr, b_prev)
    b_prev, b_curr = b_curr, res
    # accumulator = ops[op_name](accumulator, b0)

  if True:
    """
    b0 was implicitly equal to the target bit from the previous vector (for the purposes
    of calculating the current target bit), but was always set to zero in the input vector. 
    """
    input_vec[0] = 0
  
  target = b_curr
  return input_vec, target

def generate_logic_task_sequence(seq_len, num_ops):
  """
  Generate a batch of examples for the ACT logic task.
  
  Returns:
    inputs: a tensor of shape (seq_len, 102) of type torch.float.
    targets: a tensor of shape (seq_len, 1) of type torch.float.
  """
  inputs = []
  targets = []
  for i in range(seq_len):
    b0 = random.randint(0, 1) if i == 0 else int(targets[-1][0])
    inp, tgt = generate_logic_task_examplar(b0, num_ops)
    inputs.append(inp)
    targets.append([float(tgt)])  # wrap target so that its shape is (1,)
    
  inputs = torch.tensor(inputs, dtype=torch.float)
  targets = torch.tensor(targets, dtype=torch.float)
  return inputs, targets

if __name__ == '__main__':
  seq_len = 2
  num_ops = 2
  inputs, targets = generate_logic_task_sequence(seq_len, num_ops)
  print("Input examples (each row is a 102-dimensional vector):")
  print(inputs, inputs.shape)
  print("Targets:")
  print(targets, targets.shape)
