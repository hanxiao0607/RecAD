# models/tranad.py
import math
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data_utils

# ---------- Positional encoding ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, K, d_model)
        return x + self.pe[:, :x.size(1), :]

# ---------- Core Transformer AE ----------
class _TranADCore(nn.Module):
    """
    Transformer autoencoder over windows of shape (K, p).
    Reconstruction target = input; anomaly score = MSE.
    """
    def __init__(self,
                 p: int,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 use_causal_mask: bool = False):
        super().__init__()
        self.p = p
        self.d_model = d_model
        self.in_proj = nn.Linear(p, d_model)
        self.pos = PositionalEncoding(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_decoder_layers)
        self.out_proj = nn.Linear(d_model, p)
        self.use_causal_mask = use_causal_mask

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        m = torch.triu(torch.ones(T, T, device=device), diagonal=1)
        return m.masked_fill(m == 1, float("-inf"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, K, p) -> recon (B, K, p)
        """
        B, K, _ = x.shape
        src = self.in_proj(x)          # (B,K,d)
        src = self.pos(src)
        mem = self.encoder(src)        # (B,K,d)

        tgt = src                      # vanilla AE: decode from same sequence
        tgt_mask = self._causal_mask(K, x.device) if self.use_causal_mask else None
        dec = self.decoder(tgt=tgt, memory=mem, tgt_mask=tgt_mask)  # (B,K,d)
        out = self.out_proj(dec)                                    # (B,K,p)
        return out

# ---------- Public model with USAD-like surface ----------
class TranADModel(nn.Module):
    """
    Signature-compatible with UsadModel(w_size, z_size, device).
    - w_size = K * p
    - z_size is ignored (kept for compatibility)
    """
    def __init__(self, w_size: int, z_size: int, device, *,
                 K: int = None,
                 p: int = None,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 use_causal_mask: bool = False):
        super().__init__()
        self.device = device
        if K is None or p is None:
            # infer (K, p) from w_size; you must pass one of them
            raise ValueError("Please pass K and p to TranADModel for (K, p) shape.")
        assert K * p == w_size, f"w_size must equal K*p, got {w_size} vs {K}*{p}"
        self.K = K
        self.p = p
        self.core = _TranADCore(
            p=p, d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_causal_mask=use_causal_mask
        )

    # keep parity with UsadModel methods used in your pipeline
    def forward(self, w_flatten: torch.Tensor) -> torch.Tensor:
        """
        Accepts flattened windows (B, K*p) and returns reconstructions flattened (B, K*p).
        """
        if w_flatten.ndim != 2 or w_flatten.size(1) != self.K * self.p:
            raise ValueError(f"Expected (B,{self.K*self.p}) flattened input.")
        x = w_flatten.view(-1, self.K, self.p)
        y = self.core(x)
        return y.reshape(-1, self.K * self.p)

# ---------- Helpers to match usad.training/testing API ----------
@torch.no_grad()
def _batch_score(model: TranADModel, batch_flat: torch.Tensor) -> torch.Tensor:
    """
    Mean MSE over time & features per window.
    Input: (B, K*p) float on device.
    Returns: (B,) float tensor (scores)
    """
    K, p = model.K, model.p
    x = batch_flat.view(-1, K, p)
    recon = model.core(x)
    mse = ((recon - x) ** 2).mean(dim=(1, 2))
    return mse

def evaluate(model: TranADModel, val_loader, n=None, device='cpu'):
    # Kept for parity with your usad.evaluate usage (n is unused)
    model.eval()
    loss_sum, count = 0.0, 0
    with torch.no_grad():
        for [batch] in val_loader:
            batch = batch.to(device)
            loss_sum += _batch_score(model, batch).sum().item()
            count += batch.size(0)
    return {'val_loss1': loss_sum / max(1, count), 'val_loss2': loss_sum / max(1, count)}

def training(epochs, model: TranADModel, train_loader, val_loader,
             opt_func=torch.optim.Adam, device='cpu',
             lr: float = 1e-3, weight_decay: float = 0.0, grad_clip: float = 1.0):
    """
    Mirrors usad.training signature. Returns a list of val metrics per epoch.
    """
    history = []
    model.to(device)
    model.train()
    # single optimizer on the whole core (no two decoders like USAD)
    optimizer = opt_func(model.parameters(), lr=lr, weight_decay=weight_decay)
    best = float('inf')
    best_state = None

    for epoch in range(epochs):
        for [batch] in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            # reconstruction loss
            loss = _batch_score(model, batch).mean()
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            val = evaluate(model, val_loader, device=device)
        # simple logging parity
        print(f"Epoch [{epoch}], val_loss1: {val['val_loss1']:.4f}, val_loss2: {val['val_loss2']:.4f}")
        history.append(val)
        # checkpoint best
        if val['val_loss1'] < best:
            best = val['val_loss1']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.train()

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return history

@torch.no_grad()
def testing(model: TranADModel, test_loader, alpha=.5, beta=.5, device='cpu'):
    """
    Mirrors usad.testing: returns a list of per-batch score tensors.
    alpha/beta retained for signature compatibility (unused).
    """
    results = []
    model.eval()
    for [batch] in test_loader:
        batch = batch.to(device)
        scores = _batch_score(model, batch)   # (B,)
        results.append(scores)
    return results

@torch.no_grad()
def testing_sample(model: TranADModel, sample, alpha=.5, beta=.5):
    """
    Mirrors usad.testing_sample: sample is a tensor (1, K*p) on any device.
    Returns a scalar tensor (shape (1,)) with the anomaly score.
    """
    if sample.ndim == 1:
        sample = sample.unsqueeze(0)
    scores = _batch_score(model, sample.to(next(model.parameters()).device))
    return scores  # (1,)
