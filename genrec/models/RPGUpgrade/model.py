"""
RPGUpgrade: RPG with end-to-end Differentiable Product Quantization (DPQ)

Key differences from RPG:
  - Input embeddings come from a frozen sentence embedding table (not GPT-2 wte)
  - A learnable DPQ module (with Gumbel-Softmax + STE) sits between sentence
    embeddings and GPT-2, making the quantization step end-to-end trainable
  - The DPQ module contains:
        R  – orthogonal rotation matrix (warm-init from FAISS OPQ, stays orthogonal)
        K  – Key codebooks for similarity (warm-init from FAISS PQ centroids)
        V  – Value codebooks for reconstruction (separate from K)
  - Everything downstream of DPQ (GPT-2, prediction heads, loss, generate) is
    identical to RPG
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import orthogonal
from transformers import GPT2Config, GPT2Model

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer


# ---------------------------------------------------------------------------
# DPQ Module
# ---------------------------------------------------------------------------

class DPQ(nn.Module):
    """
    Differentiable Product Quantization.

    Maps (B, L, d) continuous embeddings through a discrete bottleneck and
    returns (B, L, D * v_dim) reconstructed embeddings via STE, so that
    gradients flow back through K, V, and R.

    Pipeline per forward call:
        1. Rotate:   x_rot  = x @ R^T                     (B, L, d)
        2. Split:    x_sub  = x_rot split into D chunks    (B, L, D, d/D)
        3. Logits:   l      = x_sub · K^T                 (B, L, D, n_clusters)
        4. Sample:   C̃      = Gumbel-Softmax(l, τ)        (B, L, D, n_clusters)
        5. Hard:     codes  = argmax(l)                    (B, L, D)  [no grad]
        6. Hard rec: H_hard = V[codes]                     (B, L, D, v_dim)
        7. Soft rec: H_soft = C̃ @ V                       (B, L, D, v_dim)
        8. STE:      H      = H_hard + H_soft - sg(H_soft) (B, L, D, v_dim)
        9. Flatten:  output = H.reshape(B, L, D * v_dim)

    Args:
        d          : Sentence embedding dimension (input).
        D          : Number of PQ subspaces (= n_codebook in config).
        n_clusters : Codebook size K per subspace.
        v_dim      : Value vector dimension per subspace.
                     Output dimension = D * v_dim.
        tokenizer  : RPGUpgradeTokenizer; used for warm-initialisation of K.
                     (R is initialised as a random orthogonal matrix — see note.)

    Note on R warm-init:
        PyTorch's `orthogonal` parametrisation applies a non-linear map
        (Cayley / Householder) to an internal `original` parameter, so there is
        no straightforward way to back-solve a specific target orthogonal matrix.
        R therefore starts as a *random* orthogonal matrix.  K and V are
        warm-initialised from the FAISS PQ centroids when available.
    """

    def __init__(self, d: int, D: int, n_clusters: int, v_dim: int, tokenizer):
        super().__init__()
        assert d % D == 0, f"Sentence embedding dim ({d}) must be divisible by D ({D})"

        self.d = d
        self.D = D
        self.n_clusters = n_clusters
        self.sub_dim = d // D
        self.v_dim = v_dim

        # --- Learnable orthogonal rotation R ---------------------------------
        # nn.Linear weight shape: (d_out, d_in).  forward: y = x @ weight^T
        # so weight = R means y = x @ R^T, matching FAISS OPQ convention.
        self.rotation = nn.Linear(d, d, bias=False)
        orthogonal(self.rotation)   # constrains rotation.weight to be orthogonal

        # --- Key codebooks K (D, n_clusters, sub_dim) ------------------------
        if tokenizer.pq_codebooks is not None:
            K_init = torch.from_numpy(tokenizer.pq_codebooks).float()  # (D, K, sub_dim)
        else:
            K_init = torch.randn(D, n_clusters, self.sub_dim) * 0.02
        self.K = nn.Parameter(K_init)

        # --- Value codebooks V (D, n_clusters, v_dim) ------------------------
        if v_dim == self.sub_dim and tokenizer.pq_codebooks is not None:
            V_init = torch.from_numpy(tokenizer.pq_codebooks).float()
        else:
            V_init = torch.randn(D, n_clusters, v_dim) * 0.02
        self.V = nn.Parameter(V_init)

    def forward(self, x: torch.Tensor, tau: float = 1.0) -> dict:
        """
        Args:
            x   : (B, L, d) sentence embeddings
            tau : Gumbel-Softmax temperature (lower = harder assignments)

        Returns a dict with keys:
            'ste'   : (B, L, D * v_dim)  STE output used for downstream layers
            'soft'  : (B, L, D * v_dim)  soft reconstruction (gradient path)
            'hard'  : (B, L, D * v_dim)  hard reconstruction (forward value)
            'codes' : (B, L, D)          discrete code indices (long tensor)
        """
        B, L, _ = x.shape

        # 1. Rotate
        x_rot = self.rotation(x)                              # (B, L, d)

        # 2. Split into D subspaces
        x_sub = x_rot.view(B, L, self.D, self.sub_dim)       # (B, L, D, sub_dim)

        # 3. Similarity logits against Key codebooks
        # x_sub: (B, L, D, sub_dim)  K: (D, n_clusters, sub_dim)
        logits = torch.einsum('bldi,dki->bldk', x_sub, self.K)  # (B, L, D, n_clusters)

        # 4. Gumbel-Softmax during training; plain softmax during inference
        if self.training:
            gumbel = -torch.log(
                -torch.log(torch.rand_like(logits).clamp(min=1e-10)) + 1e-10
            )
            soft_probs = F.softmax((logits + gumbel) / tau, dim=-1)
        else:
            soft_probs = F.softmax(logits / tau, dim=-1)      # (B, L, D, n_clusters)

        # 5. Hard codes (no gradient)
        codes = logits.argmax(dim=-1)                         # (B, L, D)

        # 6. Hard reconstruction: index into V
        #    V: (D, n_clusters, v_dim) → expand to (B, L, D, n_clusters, v_dim)
        V_exp = self.V.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1, -1)
        idx = codes.unsqueeze(-1).unsqueeze(-1).expand(B, L, self.D, 1, self.v_dim)
        hard = V_exp.gather(dim=3, index=idx).squeeze(3)     # (B, L, D, v_dim)

        # 7. Soft reconstruction: weighted sum of V rows
        soft = torch.einsum('bldk,dkv->bldv', soft_probs, self.V)  # (B, L, D, v_dim)

        # 8. Straight-Through Estimator
        ste = hard + soft - soft.detach()                     # (B, L, D, v_dim)

        # 9. Flatten subspace dimension
        return {
            'ste':   ste.reshape(B, L, self.D * self.v_dim),
            'soft':  soft.reshape(B, L, self.D * self.v_dim),
            'hard':  hard.reshape(B, L, self.D * self.v_dim),
            'codes': codes,
        }


# ---------------------------------------------------------------------------
# Shared building block (same as RPG)
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Residual block: x + SiLU(Linear(x)).  Initialized as identity."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.linear(x))


# ---------------------------------------------------------------------------
# RPGUpgrade model
# ---------------------------------------------------------------------------

class RPGUpgrade(AbstractModel):
    """
    GPT-2 Sequential Recommendation with end-to-end Differentiable PQ.

    Compared to RPG the only change is in how input embeddings are produced:
        RPG       : item_ids → item2tokens (frozen) → GPT-2 wte → mean-pool → GPT-2
        RPGUpgrade: item_ids → frozen sent_emb_table → DPQ (learnable) → proj → GPT-2

    Everything downstream (prediction heads, loss, generate) is identical.

    Config keys specific to RPGUpgrade:
        dpq_v_dim (int, optional): Value vector dimension per subspace.
            Defaults to n_embd // n_codebook, so DPQ output = n_embd directly
            and no projection layer is needed.
        quantizer_temperature (float): Initial Gumbel-Softmax temperature τ.
    """

    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer,
    ):
        super().__init__(config, dataset, tokenizer)

        # ------------------------------------------------------------------
        # Frozen sentence embedding table  (item_id → sent_emb)
        # ------------------------------------------------------------------
        sent_embs_tensor = torch.from_numpy(tokenizer.sent_embs)   # (n_items, d)
        self.sent_emb_table = nn.Embedding.from_pretrained(
            sent_embs_tensor, freeze=True, padding_idx=0
        )
        self.sent_emb_dim: int = sent_embs_tensor.shape[1]         # d

        # ------------------------------------------------------------------
        # Differentiable PQ module
        # ------------------------------------------------------------------
        v_dim = config.get('dpq_v_dim', config['n_embd'] // config['n_codebook'])
        self.dpq = DPQ(
            d=self.sent_emb_dim,
            D=config['n_codebook'],
            n_clusters=config['codebook_size'],
            v_dim=v_dim,
            tokenizer=tokenizer,
        )
        dpq_out_dim = config['n_codebook'] * v_dim  # = n_embd when using default

        # Optional projection if DPQ output dim ≠ GPT-2 hidden dim
        if dpq_out_dim != config['n_embd']:
            self.input_proj: nn.Module = nn.Linear(dpq_out_dim, config['n_embd'])
        else:
            self.input_proj = nn.Identity()

        # ------------------------------------------------------------------
        # GPT-2 backbone
        # ------------------------------------------------------------------
        gpt2config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            n_positions=tokenizer.max_token_seq_len,
            n_embd=config['n_embd'],
            n_layer=config['n_layer'],
            n_head=config['n_head'],
            n_inner=config['n_inner'],
            activation_function=config['activation_function'],
            resid_pdrop=config['resid_pdrop'],
            embd_pdrop=config['embd_pdrop'],
            attn_pdrop=config['attn_pdrop'],
            layer_norm_epsilon=config['layer_norm_epsilon'],
            initializer_range=config['initializer_range'],
            eos_token_id=tokenizer.eos_token,
        )
        self.gpt2 = GPT2Model(gpt2config)

        # ------------------------------------------------------------------
        # Prediction heads  (one ResBlock per digit, identical to RPG)
        # ------------------------------------------------------------------
        self.n_pred_head = tokenizer.n_digit
        self.pred_heads = nn.Sequential(
            *[ResBlock(config['n_embd']) for _ in range(self.n_pred_head)]
        )

        # ------------------------------------------------------------------
        # Item-id → 32-digit token lookup  (for loss & generate, same as RPG)
        # ------------------------------------------------------------------
        self.item_id2tokens = self._map_item_tokens().to(config['device'])

        # ------------------------------------------------------------------
        # Loss & generation parameters
        # ------------------------------------------------------------------
        self.temperature = config['temperature']
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=tokenizer.ignored_label)
        self.generate_w_decoding_graph = False
        self.init_flag = False
        self.chunk_size = config['chunk_size']
        self.num_beams = config['num_beams']
        self.n_edges = config['n_edges']
        self.propagation_steps = config['propagation_steps']

        # Gumbel temperature (caller may anneal this during training)
        self.gumbel_tau: float = config.get('quantizer_temperature', 1.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _map_item_tokens(self) -> torch.Tensor:
        """Create lookup table: item_id → 32-digit semantic code (same as RPG)."""
        item_id2tokens = torch.zeros(
            (self.dataset.n_items, self.tokenizer.n_digit), dtype=torch.long
        )
        for item in self.tokenizer.item2tokens:
            item_id = self.dataset.item2id[item]
            item_id2tokens[item_id] = torch.LongTensor(self.tokenizer.item2tokens[item])
        return item_id2tokens

    @property
    def n_parameters(self) -> str:
        total   = sum(p.numel() for p in self.parameters()      if p.requires_grad)
        dpq_p   = sum(p.numel() for p in self.dpq.parameters()  if p.requires_grad)
        gpt2_p  = sum(p.numel() for p in self.gpt2.parameters() if p.requires_grad)
        return (
            f'#DPQ parameters:   {dpq_p}\n'
            f'#GPT-2 parameters: {gpt2_p}\n'
            f'#Total trainable:  {total}\n'
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: dict, return_loss: bool = True):
        """
        1. Look up frozen sentence embeddings for each item in the sequence.
        2. Pass through DPQ → STE output feeds into GPT-2.
        3. Apply prediction heads.
        4. Optionally compute cross-entropy loss over labeled positions.
        """

        # 1. Frozen sentence embeddings
        sent_embs = self.sent_emb_table(batch['input_ids'])    # (B, L, d)

        # 2. Differentiable quantization
        dpq_out = self.dpq(sent_embs, tau=self.gumbel_tau)
        input_embs = self.input_proj(dpq_out['ste'])            # (B, L, n_embd)

        # 3. GPT-2 encoding
        outputs = self.gpt2(
            inputs_embeds=input_embs,
            attention_mask=batch['attention_mask'],
        )

        # 4. Prediction heads  (n_pred_head × ResBlock)
        final_states = torch.cat(
            [self.pred_heads[i](outputs.last_hidden_state).unsqueeze(-2)
             for i in range(self.n_pred_head)],
            dim=-2,
        )                                                       # (B, L, D, n_embd)
        outputs.final_states = final_states

        # 5. Loss (identical to RPG)
        if return_loss:
            assert 'labels' in batch, 'Batch must contain labels.'

            label_mask = batch['labels'].view(-1) != -100

            selected = final_states.view(
                -1, self.n_pred_head, self.config['n_embd']
            )[label_mask]                                       # (N_valid, D, n_embd)
            selected = F.normalize(selected, dim=-1)
            selected = torch.chunk(selected, self.n_pred_head, dim=1)

            token_emb = F.normalize(self.gpt2.wte.weight[1:-1], dim=-1)
            token_embs = torch.chunk(token_emb, self.n_pred_head, dim=0)

            token_logits = [
                torch.matmul(selected[i].squeeze(1), token_embs[i].T) / self.temperature
                for i in range(self.n_pred_head)
            ]
            token_labels = self.item_id2tokens[batch['labels'].view(-1)[label_mask]]

            losses = [
                self.loss_fct(
                    token_logits[i],
                    token_labels[:, i] - i * self.config['codebook_size'] - 1,
                )
                for i in range(self.n_pred_head)
            ]
            outputs.loss = torch.mean(torch.stack(losses))

        return outputs

    # ------------------------------------------------------------------
    # Generate  (identical logic to RPG)
    # ------------------------------------------------------------------

    def generate(self, batch: dict, n_return_sequences: int = 1, return_loss: bool = False):
        """
        Predict top-k next items.

        Args:
            batch               : Input batch dict.
            n_return_sequences  : How many top items to return.
            return_loss         : If True, also return the validation loss.

        Returns:
            preds               : (B, n_return_sequences, 1) item IDs
            loss (optional)     : scalar tensor, only when return_loss=True
        """
        outputs = self.forward(batch, return_loss=return_loss)

        # Extract hidden state at the last valid position for each example
        states = outputs.final_states.gather(
            dim=1,
            index=(batch['seq_lens'] - 1).view(-1, 1, 1, 1).expand(
                -1, 1, self.n_pred_head, self.config['n_embd']
            ),
        )                                                       # (B, 1, D, n_embd)
        states = F.normalize(states, dim=-1)

        token_emb = F.normalize(self.gpt2.wte.weight[1:-1], dim=-1)
        token_embs = torch.chunk(token_emb, self.n_pred_head, dim=0)

        logits = [
            F.log_softmax(
                torch.matmul(states[:, 0, i, :], token_embs[i].T) / self.temperature,
                dim=-1,
            )
            for i in range(self.n_pred_head)
        ]
        token_logits = torch.cat(logits, dim=-1)               # (B, D * codebook_size)

        # Score every item by summing log-probs across its 32 digits
        item_logits = torch.gather(
            input=token_logits.unsqueeze(-2).expand(-1, self.dataset.n_items, -1),
            dim=-1,
            index=(self.item_id2tokens[1:, :] - 1)
                  .unsqueeze(0).expand(token_logits.shape[0], -1, -1),
        ).mean(dim=-1)                                          # (B, n_items)

        preds = item_logits.topk(n_return_sequences, dim=-1).indices + 1
        preds = preds.unsqueeze(-1)                             # (B, n_return_sequences, 1)

        if return_loss:
            return preds, outputs.loss
        return preds
