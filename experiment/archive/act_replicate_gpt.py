# act_logic.py
import math, torch, torch.nn as nn, torch.nn.functional as F

# ---------- hyper-params ----------
H      = 128      # hidden size
EPS    = 1e-2     # Graves' halting threshold ε
MAX_H  = 10       # upper bound on ponder steps N
LR     = 1e-3
BATCH  = 128
SEQ    = 64       # 64 logic triples → 64 outputs
STEPS  = 30_000   # training iterations
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------- data generator ----------
def gen_batch(bs=BATCH, seq=SEQ, device=DEVICE):
    # 0-3  →  AND OR XOR NAND   (XNOR & friends omitted for brevity)
    op   = torch.randint(0, 4, (bs, seq, 1), device=device)
    a, b = torch.randint(0, 2, (bs, seq, 1), device=device), torch.randint(0, 2,(bs,seq,1), device=device)

    # one-hot op (2 bits are enough for 4 gates)
    op_oh = F.one_hot(op.squeeze(-1), 4).float()

    x  = torch.cat([op_oh, a, b], dim=-1)        # (bs, seq, 6)
    y  = torch.zeros_like(a)                     # allocation (bs,seq,1)

    # ground-truth
    y[op.eq(0)] = a[op.eq(0)] &  b[op.eq(0)]     # AND
    y[op.eq(1)] = a[op.eq(1)] |  b[op.eq(1)]     # OR
    y[op.eq(2)] = a[op.eq(2)] ^  b[op.eq(2)]     # XOR
    y[op.eq(3)] = ~(a[op.eq(3)] & b[op.eq(3)]) & 1 # NAND

    return x, y.squeeze(-1).long()

# ---------- ACT layer ----------
class ACTCell(nn.Module):
    def __init__(self, inp, hid, eps=EPS, max_h=MAX_H):
        super().__init__()
        self.cell   = nn.GRUCell(inp, hid)
        self.hid2p  = nn.Linear(hid, 1)          # halting unit
        self.hid2y  = nn.Linear(hid, 2)          # binary classification
        self.eps, self.max_h = eps, max_h

    def forward(self, x):                        # x: (bs, seq, inp)
        bs, seq, _ = x.shape
        h    = torch.zeros(bs, self.cell.hidden_size, device=x.device)
        logits = []
        ponder_cost = 0.0

        for t in range(seq):
            # (bs,) accumulators
            halting_prob = torch.zeros(bs, device=x.device)
            remainders   = torch.zeros(bs, device=x.device)
            n_updates    = torch.zeros(bs, device=x.device)
            still_running = torch.ones(bs, dtype=torch.bool, device=x.device)

            y_t = torch.zeros(bs, 2, device=x.device)  # final mixture

            step = 0
            while still_running.any() and step < self.max_h:
                h = self.cell(x[:, t], h)              # one micro-step
                p = torch.sigmoid(self.hid2p(h)).squeeze(-1)

                # decide which examples halt this step
                new_halt = still_running & (halting_prob + p > 1 - self.eps)
                cont     = still_running & ~new_halt

                # probability mass contributed this step
                add_prob = torch.where(new_halt, 1 - halting_prob, p)
                halting_prob = halting_prob + add_prob
                remainders   = torch.where(new_halt, add_prob, remainders)
                still_running = cont
                n_updates    = n_updates + (~new_halt).float()

                # accumulate output
                y_t = y_t + add_prob.unsqueeze(-1) * self.hid2y(h)
                step += 1

            ponder_cost += n_updates.mean()
            logits.append(y_t)

        return torch.stack(logits, dim=1), ponder_cost / seq  # (bs,seq,2), scalar

# ---------- tiny classifier ----------
model = ACTCell(inp=6, hid=H).to(DEVICE)
opt   = torch.optim.Adam(model.parameters(), lr=LR)

# ---------- training loop ----------
for it in range(1, STEPS+1):
    x, y = gen_batch()
    logit, ponder = model(x)

    loss_task   = F.cross_entropy(logit.view(-1, 2), y.view(-1))
    loss        = loss_task + ponder * 1e-3          # τ = 0.001
    opt.zero_grad(); loss.backward(); opt.step()

    pred = logit.argmax(-1)
    acc  = (pred == y).float().mean()
    print(f"(train) it {it:>5} | loss {loss.item():.4f} | acc {acc*100:5.2f}% | ponder {ponder.item():.2f}")

    if it % 1000 == 0:
        with torch.no_grad():
            pred = logit.argmax(-1)
            acc  = (pred == y).float().mean()
            print(f"it {it:>5} | loss {loss.item():.4f} | acc {acc*100:5.2f}% | ponder {ponder.item():.2f}")
