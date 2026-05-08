# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Upgraded RPG Model with Differentiable OPQ Layer

This module implements an enhanced version of RPG (Residual Quantization Product) that:
1. Uses a Differentiable OPQ layer to learn rotation matrices and codebooks during training
2. Applies Gumbel-Softmax trick for smooth quantization during forward pass
3. Uses Straight-Through Estimator (STE) to enable backpropagation through discrete operations
4. Maintains all benefits of the original RPG while making quantization parameters learnable

Key Features:
- DifferentiableOPQ: Learnable rotation matrix R and codebooks K
- Gumbel-Softmax: Soft quantization with temperature annealing
- Straight-Through Estimator: Hard decisions in forward, soft gradients in backward
- Orthogonal Constraint: Rotation matrix stays orthogonal via parametrization
- Parallel Prediction Heads: 32 ResBlocks for predicting digit codes

Mathematical Foundation:
    Input Embedding x (batch_size, seq_len, emb_dim)
        ↓
    Rotation: y = x @ R  (R is orthogonal, stays orthogonal during training)
        ↓
    Split into D=32 subspaces: y = [y_1, y_2, ..., y_32]
        ↓
    For each subspace d:
        - Compute similarities: s_d = y_d @ K_d^T  (shape: batch, seq_len, codebook_size=256)
        - Apply Gumbel-Softmax: p_d = softmax((s_d + gumbel_noise) / tau)
        - Hard assignment: z_d = argmax(p_d)
        - Soft reconstruction: h_d = p_d @ K_d
        - STE output: h_d + sg(z_d - h_d)
        ↓
    Concatenate: H = [h_1, h_2, ..., h_32]  (batch_size, seq_len, emb_dim)
        ↓
    Feed to GPT-2 backbone
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer


class GumbelSoftmax(nn.Module):
    """
    Gumbel-Softmax layer for differentiable sampling.
    
    Converts logits to soft probabilities in a differentiable way using the Gumbel trick.
    During training, outputs are soft; during inference (deterministic mode), outputs are hard.
    
    Mathematical Formulation:
        Soft output: softmax((log_alpha + G) / tau)
        Hard output: one-hot(argmax(log_alpha))
        Straight-Through: output + sg(hard - soft)
    
    where:
        - log_alpha: log-probabilities (logits)
        - G: Gumbel noise sampled from Gumbel(0,1)
        - tau: temperature (high tau = soft, low tau = hard)
        - sg: stop-gradient operator
    
    Temperature Annealing Strategy (optional):
        Start with high temperature for exploration, gradually decrease for exploitation.
        Use in training loop: gumbel_softmax.set_temperature(epoch / max_epochs)
    """
    
    def __init__(self, temperature: float = 1.0, hard: bool = True):
        """
        Initialize Gumbel-Softmax layer.
        
        Args:
            temperature: Initial temperature (default 1.0).
                        Higher values produce softer distributions.
                        Lower values produce harder (near one-hot) distributions.
            hard: If True, use straight-through estimator to pass hard decisions
                 while maintaining gradient flow through soft probabilities.
        """
        super().__init__()
        self.temperature = temperature
        self.hard = hard

    def set_temperature(self, temperature: float):
        """Update temperature for annealing schedule."""
        self.temperature = max(temperature, 1e-3)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply Gumbel-Softmax transformation.
        
        Args:
            logits: Log-probabilities of shape (..., num_classes).
                   Can have arbitrary leading dimensions.
        
        Returns:
            outputs: 
                - If training: soft probabilities (differentiable, sum to 1)
                - If eval: hard one-hot vectors
                Shape matches input: (..., num_classes)
        
        Example:
            >>> gs = GumbelSoftmax(temperature=0.5)
            >>> logits = torch.randn(4, 10, 256)  # batch=4, seq_len=10, codebook_size=256
            >>> soft_probs = gs(logits)
            >>> soft_probs.shape
            torch.Size([4, 10, 256])
            >>> soft_probs.sum(dim=-1)  # Should be all 1.0
            tensor([[[1., 1., ...],
                     ...]])
        """
        if self.training:
            # Sample Gumbel noise
            # Gumbel(0,1) = -log(-log(Uniform(0,1)))
            u = torch.rand_like(logits)
            gumbel_noise = -torch.log(-torch.log(u + 1e-20) + 1e-20)
            
            # Soft probabilities with temperature
            y_soft = F.softmax((logits + gumbel_noise) / self.temperature, dim=-1)
            
            if self.hard:
                # Hard decision (one-hot)
                y_hard = F.one_hot(y_soft.argmax(dim=-1), num_classes=logits.size(-1)).float()
                # Straight-Through Estimator: use hard in forward, soft in backward
                # This allows backprop to flow through the soft distribution
                y = y_hard - y_soft.detach() + y_soft
            else:
                y = y_soft
        else:
            # Inference: use hard one-hot decisions
            y = F.one_hot(logits.argmax(dim=-1), num_classes=logits.size(-1)).float()
        
        return y


class DifferentiableOPQ(nn.Module):
    """
    Differentiable OPQ (Orthogonal Product Quantization) Layer.
    
    This layer makes the OPQ quantization process learnable through gradient descent.
    
    Architecture:
        Input: x (batch_size, seq_len, embedding_dim)
            ↓
        Rotation: y = x @ R  (R stays orthogonal via Householder reflection)
            ↓
        Subspace Split: [y_1, y_2, ..., y_D] where each y_d has dim (emb_dim / D)
            ↓
        For each subspace d:
            - Similarity scores: s_d = y_d @ K_d^T  (shape: batch, seq_len, K)
            - Gumbel-Softmax: p_d = GumbelSoftmax(s_d)
            - Soft reconstruction: h_d = p_d @ K_d
            - Hard assignment: z_d = argmax(p_d)  (discrete token)
            - STE output: h_d + sg(z_d - h_d)
            ↓
        Concatenate reconstructions: H = [h_1, h_2, ..., h_D]
        Stack hard assignments: Z = [z_1, z_2, ..., z_D]  (shape: batch, seq_len, D)
            ↓
        Return: (H, Z) - reconstructed embeddings and discrete tokens
    
    Learned Parameters:
        - rotation_matrix: R (embedding_dim, embedding_dim) - orthogonal
        - codebooks: K_d for each subspace (D, K, emb_dim/D)
    
    Key Innovation:
        - Straight-Through Estimator allows backpropagation through discrete argmax
        - Gumbel-Softmax provides smooth approximation during training
        - Orthogonal constraint on R via torch.nn.utils.parametrizations.orthogonal
        - Temperature annealing: start soft (high tau) → end hard (low tau)
    
    Tensor Shape Examples (with embedding_dim=512, n_codebook=32, codebook_size=256):
        - Input x:          (batch=4, seq_len=20, 512)
        - After rotation y: (batch=4, seq_len=20, 512)
        - Per subspace y_d: (batch=4, seq_len=20, 16)  where 16 = 512/32
        - Similarities s_d: (batch=4, seq_len=20, 256)
        - Soft probs p_d:   (batch=4, seq_len=20, 256)
        - Codebook K_d:     (256, 16)
        - Reconstruction h_d: (batch=4, seq_len=20, 16)
        - Hard tokens z_d:  (batch=4, seq_len=20) with values in [0, 255]
        - Final output H:   (batch=4, seq_len=20, 512)
        - Final output Z:   (batch=4, seq_len=20, 32)
    """
    
    def __init__(
        self,
        embedding_dim: int,
        n_codebook: int,
        codebook_size: int,
        rotation_matrix: torch.Tensor = None,
        codebook_matrices: torch.Tensor = None,
        temperature: float = 1.0,
        learn_temperature: bool = False
    ):
        """
        Initialize DifferentiableOPQ layer.
        
        Args:
            embedding_dim: Dimensionality of input embeddings (e.g., 512).
            n_codebook: Number of subspaces (typically 32).
            codebook_size: Number of centroids per subspace (typically 256).
            rotation_matrix: Pre-trained OPQ rotation matrix (embedding_dim, embedding_dim).
                           If None, initialized as identity + small random values.
            codebook_matrices: Pre-trained PQ codebooks (n_codebook, codebook_size, embedding_dim//n_codebook).
                              If None, randomly initialized.
            temperature: Initial Gumbel-Softmax temperature (1.0 = default).
            learn_temperature: If True, temperature becomes learnable parameter (usually False).
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.n_codebook = n_codebook
        self.codebook_size = codebook_size
        self.subspace_dim = embedding_dim // n_codebook
        
        assert embedding_dim % n_codebook == 0, \
            f"embedding_dim ({embedding_dim}) must be divisible by n_codebook ({n_codebook})"
        
        # ============ Initialize Rotation Matrix ============
        if rotation_matrix is not None:
            # Initialize with extracted OPQ rotation matrix
            # Ensure it's orthogonal using Householder reflection parametrization
            rotation_init = rotation_matrix.to(torch.float32)
        else:
            # Initialize as identity + small random perturbation
            rotation_init = torch.eye(embedding_dim) + 0.01 * torch.randn(embedding_dim, embedding_dim)
            # Orthogonalize via QR decomposition
            rotation_init, _ = torch.linalg.qr(rotation_init)
        
        # Register as a parametrized linear layer with orthogonal constraint
        self.rotation = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.rotation.weight.data = rotation_init.T  # Linear uses weight @ x^T convention
        
        # Apply orthogonal parametrization to maintain orthogonality during training
        self.rotation = torch.nn.utils.parametrizations.orthogonal(self.rotation)
        
        # ============ Initialize Codebooks ============
        if codebook_matrices is not None:
            # Use pre-trained codebooks from OPQ
            codebook_init = codebook_matrices.to(torch.float32)
        else:
            # Random initialization: each codebook entry is random in subspace
            codebook_init = torch.randn(n_codebook, codebook_size, self.subspace_dim) * 0.01
            # Normalize codebook entries to unit norm (improves numerical stability)
            codebook_init = F.normalize(codebook_init, dim=-1)
        
        # Register codebooks as learnable parameters
        self.codebooks = nn.Parameter(codebook_init)
        
        # ============ Gumbel-Softmax Configuration ============
        self.gumbel_softmax = GumbelSoftmax(temperature=temperature, hard=True)
        
        # Temperature annealing (optional)
        if learn_temperature:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer('temperature', torch.tensor(temperature))
    
    def forward(
        self,
        embeddings: torch.Tensor,
        return_soft: bool = True,
        return_hard: bool = True
    ) -> dict:
        """
        Apply Differentiable OPQ quantization to embeddings.
        
        Args:
            embeddings: Input embeddings of shape (batch_size, seq_len, embedding_dim).
            return_soft: If True, return soft reconstructed embeddings (for backprop).
            return_hard: If True, return hard discrete tokens (for transformer input).
        
        Returns:
            dict with keys:
                - 'soft': Soft reconstructed embeddings (batch_size, seq_len, embedding_dim)
                - 'hard': Hard discrete tokens (batch_size, seq_len, n_codebook) with values in [0, codebook_size-1]
                - 'logits': Similarity scores before softmax (batch_size, seq_len, n_codebook, codebook_size)
                - 'probabilities': Gumbel-Softmax probabilities (batch_size, seq_len, n_codebook, codebook_size)
        
        Example:
            >>> dopq = DifferentiableOPQ(512, 32, 256)
            >>> embeddings = torch.randn(4, 20, 512)
            >>> outputs = dopq(embeddings)
            >>> outputs['soft'].shape
            torch.Size([4, 20, 512])
            >>> outputs['hard'].shape
            torch.Size([4, 20, 32])
        """
        batch_size, seq_len, _ = embeddings.shape
        
        # ============ Step 1: Apply Rotation ============
        # y = x @ R  where R is orthogonal
        # Shape: (batch_size, seq_len, embedding_dim)
        rotated = self.rotation(embeddings)
        
        # ============ Step 2: Split into Subspaces ============
        # Reshape to isolate each subspace
        # (batch_size, seq_len, embedding_dim) → (batch_size, seq_len, n_codebook, subspace_dim)
        subspaces = rotated.view(batch_size, seq_len, self.n_codebook, self.subspace_dim)
        
        # ============ Step 3: Quantize Each Subspace ============
        soft_reconstructions = []
        hard_tokens = []
        all_logits = []
        all_probabilities = []
        
        for d in range(self.n_codebook):
            # Extract subspace d
            # (batch_size, seq_len, subspace_dim)
            subspace_d = subspaces[:, :, d, :]
            
            # Compute similarity scores with codebook
            # subspace_d @ codebooks[d].T
            # (batch_size, seq_len, subspace_dim) @ (subspace_dim, codebook_size)
            # = (batch_size, seq_len, codebook_size)
            logits_d = torch.matmul(subspace_d, self.codebooks[d].T)
            all_logits.append(logits_d)
            
            # Apply Gumbel-Softmax to get soft probabilities
            # Shape: (batch_size, seq_len, codebook_size)
            probs_d = self.gumbel_softmax(logits_d)
            all_probabilities.append(probs_d)
            
            # Soft reconstruction: weighted sum of codebook entries
            # probs_d @ codebooks[d]
            # (batch_size, seq_len, codebook_size) @ (codebook_size, subspace_dim)
            # = (batch_size, seq_len, subspace_dim)
            soft_recon_d = torch.matmul(probs_d, self.codebooks[d])
            soft_reconstructions.append(soft_recon_d)
            
            # Hard assignment: argmax to get discrete tokens
            # (batch_size, seq_len)
            hard_tokens_d = logits_d.argmax(dim=-1)
            hard_tokens.append(hard_tokens_d)
        
        # ============ Step 4: Apply Straight-Through Estimator ============
        # For each subspace d:
        #   output = soft_recon_d + sg(hard_recon_d - soft_recon_d)
        # This allows:
        #   - Forward pass: discrete hard decisions (deterministic)
        #   - Backward pass: gradients flow through soft probabilities
        
        final_outputs = []
        for d in range(self.n_codebook):
            # Hard reconstruction using hard tokens
            # hard_tokens[d] shape: (batch_size, seq_len)
            # codebooks[d] shape: (codebook_size, subspace_dim)
            # Result shape: (batch_size, seq_len, subspace_dim)
            hard_recon_d = self.codebooks[d][hard_tokens[d]]
            
            # Straight-Through Estimator
            # output = soft + sg(hard - soft)
            # This ensures:
            #   - Forward: hard_recon_d (discrete)
            #   - Backward: gradient of soft_recon_d (smooth)
            output_d = soft_reconstructions[d] + (hard_recon_d - soft_reconstructions[d]).detach()
            final_outputs.append(output_d)
        
        # ============ Step 5: Concatenate Subspaces ============
        # Stack all subspace reconstructions
        # List of (batch_size, seq_len, subspace_dim) → (batch_size, seq_len, n_codebook, subspace_dim)
        final_tensor = torch.stack(final_outputs, dim=2)
        
        # Reshape back to original embedding dimension
        # (batch_size, seq_len, n_codebook, subspace_dim) → (batch_size, seq_len, embedding_dim)
        soft_output = final_tensor.view(batch_size, seq_len, self.embedding_dim)
        
        # Stack hard tokens
        # List of (batch_size, seq_len) → (batch_size, seq_len, n_codebook)
        hard_output = torch.stack(hard_tokens, dim=2)
        
        # Stack logits and probabilities for analysis
        logits_tensor = torch.stack(all_logits, dim=2)  # (batch, seq_len, n_codebook, codebook_size)
        probs_tensor = torch.stack(all_probabilities, dim=2)  # (batch, seq_len, n_codebook, codebook_size)
        
        return {
            'soft': soft_output,
            'hard': hard_output,
            'logits': logits_tensor,
            'probabilities': probs_tensor
        }
    
    def set_temperature(self, temperature: float):
        """Update Gumbel-Softmax temperature for annealing."""
        self.gumbel_softmax.set_temperature(temperature)


class ResBlock(nn.Module):
    """
    Lightweight Residual Block for refining predictions.
    
    Architecture: x + SiLU(Linear(x))
    - Initialized as identity (zero weights)
    - SiLU activation adds non-linearity
    
    Input/Output shape: (..., hidden_size)
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)
        torch.nn.init.zeros_(self.linear.bias)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Residual connection: x + SiLU(Linear(x))."""
        return x + self.act(self.linear(x))


class RPGUpgrade(AbstractModel):
    """
    Upgraded RPG: Sequential Recommendation with Differentiable OPQ.
    
    This model extends the original RPG by making the quantization layer learnable.
    
    Architecture:
        Input Item Sequence
            ↓
        Tokenizer: item_id → 32-digit semantic code (from pre-trained OPQ)
            ↓
        DifferentiableOPQ: (optional) Learn to improve quantization during training
            - Rotation matrix R: Stays orthogonal via parametrization
            - Codebooks K_d: Learn better centroids for each subspace
            - Gumbel-Softmax: Smooth approximation to discrete quantization
            ↓
        Token Embedding Layer
            ↓
        GPT-2 Backbone: Model sequential patterns (2 transformer layers)
            ↓
        32 Parallel Prediction Heads (ResBlocks): Refine predictions for each digit
            ↓
        Loss: 32 independent cross-entropy losses (one per digit)
            ↓
        Inference: Reconstruct items from predicted digits
    
    Key Parameters (from config.yaml):
        - n_embd: 448 (hidden dimension)
        - n_layer: 2 (transformer layers)
        - n_head: 4 (attention heads)
        - n_codebook: 32 (number of digits)
        - codebook_size: 256 (values per digit)
        - temperature: 0.07 (Gumbel-Softmax temperature)
    
    Training:
        - Use Gumbel-Softmax during forward pass for smooth quantization
        - Apply temperature annealing: high temp (start) → low temp (end)
        - Use STE to pass hard decisions while maintaining gradient flow
        - Optimize: embedding weights + rotation matrix + codebooks
    
    Inference:
        - Use hard discrete tokens from argmax
        - No temperature annealing (deterministic)
        - Fast inference with integer operations
    """
    
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer,
        use_differentiable_opq: bool = True
    ):
        """
        Initialize the upgraded RPG model.
        
        Args:
            config: Configuration dictionary with model hyperparameters.
            dataset: Dataset with n_items, item2id mappings.
            tokenizer: Tokenizer with item2tokens and OPQ parameters (if available).
            use_differentiable_opq: If True, enable learnable DifferentiableOPQ layer.
                                   Requires tokenizer to have opq_rotation and pq_codebooks.
        """
        super(RPGUpgrade, self).__init__(config, dataset, tokenizer)
        from logging import getLogger
        self.logger = getLogger()
        
        self.use_differentiable_opq = use_differentiable_opq
        
        # ============ STEP 1: Create item_id → token mapping ============
        self.item_id2tokens = self._map_item_tokens().to(self.config['device'])
        
        # ============ STEP 2: Initialize Differentiable OPQ (if enabled) ============
        if use_differentiable_opq:
            if hasattr(tokenizer, 'opq_rotation') and hasattr(tokenizer, 'pq_codebooks') and \
               tokenizer.opq_rotation is not None and tokenizer.pq_codebooks is not None:
                self.dopq = DifferentiableOPQ(
                    embedding_dim=config['n_embd'],
                    n_codebook=tokenizer.n_digit,
                    codebook_size=tokenizer.codebook_size,
                    rotation_matrix=tokenizer.opq_rotation.to(self.config['device']),
                    codebook_matrices=tokenizer.pq_codebooks.to(self.config['device']),
                    temperature=config.get('quantizer_temperature', 1.0)
                ).to(self.config['device'])
                self.log(f"[MODEL] DifferentiableOPQ initialized with extracted OPQ parameters")
            else:
                self.dopq = DifferentiableOPQ(
                    embedding_dim=config['n_embd'],
                    n_codebook=tokenizer.n_digit,
                    codebook_size=tokenizer.codebook_size,
                    temperature=config.get('quantizer_temperature', 1.0)
                ).to(self.config['device'])
                self.log(f"[MODEL] DifferentiableOPQ initialized with random parameters")
        else:
            self.dopq = None
            self.log(f"[MODEL] DifferentiableOPQ disabled")
        
        # ============ STEP 3: Initialize GPT-2 backbone ============
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
        self.gpt2 = self.gpt2.to(self.config['device'])
        
        # ============ STEP 4: Create 32 parallel prediction heads ============
        self.n_pred_head = self.tokenizer.n_digit
        pred_head_list = []
        for i in range(self.n_pred_head):
            pred_head_list.append(ResBlock(self.config['n_embd']))
        self.pred_heads = nn.Sequential(*pred_head_list)
        self.pred_heads = self.pred_heads.to(self.config['device'])
        
        # ============ STEP 5: Setup loss and generation parameters ============
        self.temperature = self.config['temperature']
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.ignored_label)
        
        # Graph-constrained decoding parameters
        self.generate_w_decoding_graph = False
        self.init_flag = False
        self.chunk_size = config['chunk_size']
        self.num_beams = config['num_beams']
        self.n_edges = config['n_edges']
        self.propagation_steps = config['propagation_steps']

    def _map_item_tokens(self) -> torch.Tensor:
        """Create item_id → semantic code lookup table."""
        item_id2tokens = torch.zeros((self.dataset.n_items, self.tokenizer.n_digit), dtype=torch.long)
        for item in self.tokenizer.item2tokens:
            item_id = self.dataset.item2id[item]
            item_id2tokens[item_id] = torch.tensor(self.tokenizer.item2tokens[item], dtype=torch.long)
        return item_id2tokens

    @property
    def n_parameters(self) -> str:
        """Return model parameter count."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return f"Total: {total_params:,} | Trainable: {trainable_params:,}"

    def forward(self, batch: dict, return_loss: bool = True) -> dict:
        """
        Forward pass: Predict next item's digit codes with optional DifferentiableOPQ.
        
        TENSOR SHAPE WALKTHROUGH (example: batch_size=4, seq_len=10):
        
        INPUT:
          batch['input_ids']:      (4, 10)     - item indices from dataset
          batch['attention_mask']: (4, 10)     - 1 for valid, 0 for padding
          batch['labels']:         (4, 10)     - next item indices
        
        STEP 1 - Token Lookup:
          item_id2tokens lookup: (4, 10) → (4, 10, 32)
        
        STEP 2 - Token Embedding:
          GPT-2 embedding: (4, 10, 32) → (4, 10, 32, 768)
          Average across 32 codes: (4, 10, 768)
        
        STEP 3 - [Optional] DifferentiableOPQ:
          If enabled, apply learnable quantization layer
          Input: (4, 10, 768)
          Output: (4, 10, 768) - reconstructed embeddings with learned rotation/codebooks
        
        STEP 4 - GPT-2 Encoding:
          Process through transformer layers
          Input:  (4, 10, 768)
          Output: (4, 10, 768)
        
        STEP 5 - Prediction Heads (32 parallel ResBlocks):
          Apply 32 independent ResBlocks
          Output: (4, 10, 32, 768)
        
        STEP 6 - Loss Computation:
          Extract valid positions, normalize, compute cross-entropy per digit
        
        Args:
            batch: Dict with 'input_ids', 'attention_mask', 'labels', 'seq_lens'
            return_loss: If True, compute training loss
        
        Returns:
            outputs: Object with .loss and .final_states
        """
        
        # ============ STEP 1: Token Lookup ============
        input_tokens = self.item_id2tokens[batch['input_ids']]
        
        # ============ STEP 2: Token Embedding & Averaging ============
        input_embs = self.gpt2.wte(input_tokens).mean(dim=-2)
        
        # ============ STEP 3: [Optional] DifferentiableOPQ ============
        if self.use_differentiable_opq and self.dopq is not None:
            dopq_output = self.dopq(input_embs, return_soft=True, return_hard=True)
            # Use soft reconstructions for backprop
            input_embs = dopq_output['soft']
            # Store hard tokens for analysis (optional)
            hard_tokens = dopq_output['hard']
        
        # ============ STEP 4: GPT-2 Transformer Encoding ============
        outputs = self.gpt2(
            inputs_embeds=input_embs,
            attention_mask=batch['attention_mask']
        )
        
        # ============ STEP 5: Apply Prediction Heads ============
        final_states = [self.pred_heads[i](outputs.last_hidden_state).unsqueeze(-2)
                       for i in range(self.n_pred_head)]
        final_states = torch.cat(final_states, dim=-2)
        
        outputs.final_states = final_states
        
        # ============ STEP 6: Loss Computation ============
        if return_loss:
            # Extract labels
            labels = batch['labels']
            
            # Create mask for valid positions (not padding)
            valid_mask = (labels != self.tokenizer.ignored_label)
            
            # Reshape tensors for easier indexing
            batch_size, seq_len = labels.shape
            # final_states: (batch, seq_len, n_digit, n_embd) -> (batch*seq_len, n_digit, n_embd)
            digit_preds_flat = final_states.view(batch_size * seq_len, self.n_pred_head, self.config['n_embd'])
            # labels: (batch, seq_len) -> (batch*seq_len,)
            labels_flat = labels.view(batch_size * seq_len)
            # valid_mask: (batch, seq_len) -> (batch*seq_len,)
            valid_mask_flat = valid_mask.view(batch_size * seq_len)
            
            # Get item tokens: (batch*seq_len, n_digit)
            item_tokens_flat = self.item_id2tokens[labels_flat]
            
            # Select only valid positions
            digit_preds_valid = digit_preds_flat[valid_mask_flat]  # (valid_count, n_digit, n_embd)
            item_tokens_valid = item_tokens_flat[valid_mask_flat]  # (valid_count, n_digit)
            
            # Compute loss for each digit separately
            total_loss = 0.0
            for digit_idx in range(self.n_pred_head):
                # Extract predictions for this digit: (valid_count, n_embd)
                digit_preds = digit_preds_valid[:, digit_idx, :]
                
                # Get label tokens for this digit: (valid_count,)
                # item_id2tokens stores token IDs (1-8192), need to convert to digit values (0-255)
                # Token ID layout: digit d's values are at indices (d*256+1) to (d*256+256)
                # So: digit_value = token_id - d*256 - 1
                label_tokens = item_tokens_valid[:, digit_idx]
                label_digit_values = label_tokens - digit_idx * self.tokenizer.codebook_size - 1
                
                # Normalize predictions (L2 norm)
                digit_preds_normalized = F.normalize(digit_preds, dim=-1)
                
                # Compute logits: similarity with embedding table
                # Extract embeddings for this digit's tokens
                # Add 1 offset because token 0 is padding, token 1+ are data
                digit_start = digit_idx * self.tokenizer.codebook_size + 1
                digit_end = (digit_idx + 1) * self.tokenizer.codebook_size + 1
                digit_embeddings = self.gpt2.wte.weight[digit_start:digit_end]
                
                # Logits: (valid_count, codebook_size)
                logits = torch.matmul(digit_preds_normalized, digit_embeddings.T) / self.temperature
                
                # Cross-entropy loss - note: label_digit_values should be in range [0, codebook_size)
                digit_loss = self.loss_fct(logits, label_digit_values)
                total_loss = total_loss + digit_loss
            
            outputs.loss = total_loss / self.n_pred_head
        
        return outputs

    def build_ii_sim_mat(self):
        """Build item-item similarity matrix using semantic embeddings."""
        n_items = self.dataset.n_items
        n_digit = self.tokenizer.n_digit
        codebook_size = self.tokenizer.codebook_size

        token_embs = self.gpt2.wte.weight[1:-1].view(n_digit, codebook_size, -1)
        token_embs = F.normalize(token_embs, dim=-1)
        token_sims = torch.bmm(token_embs, token_embs.transpose(1, 2))
        token_sims_01 = 0.5 * (token_sims + 1.0)

        item_item_sim = torch.zeros((n_items, n_items), device=self.gpt2.device, dtype=torch.float32)

        for i_start in range(1, n_items, self.chunk_size):
            i_end = min(i_start + self.chunk_size, n_items)
            chunk_size = i_end - i_start

            item_ids_chunk = torch.arange(i_start, i_end, device=self.gpt2.device)
            item_tokens_chunk = self.item_id2tokens[item_ids_chunk]  # (chunk_size, n_digit)

            for j_start in range(1, n_items, self.chunk_size):
                j_end = min(j_start + self.chunk_size, n_items)
                chunk_size_j = j_end - j_start

                item_ids_chunk_j = torch.arange(j_start, j_end, device=self.gpt2.device)
                item_tokens_chunk_j = self.item_id2tokens[item_ids_chunk_j]  # (chunk_size_j, n_digit)

                sim_chunk = torch.zeros((chunk_size, chunk_size_j), device=self.gpt2.device)
                for d in range(n_digit):
                    tokens_d_i = item_tokens_chunk[:, d]  # (chunk_size,)
                    tokens_d_j = item_tokens_chunk_j[:, d]  # (chunk_size_j,)

                    # Lookup similarities: (chunk_size, chunk_size_j)
                    sim_chunk = sim_chunk + token_sims_01[d][tokens_d_i][:, tokens_d_j]

                sim_chunk = sim_chunk / n_digit
                item_item_sim[i_start:i_end, j_start:j_end] = sim_chunk

        return item_item_sim

    def build_adjacency_list(self, item_item_sim):
        """Build k-NN graph: find nearest neighbors for each item."""
        n_items = self.dataset.n_items
        adjacency = torch.zeros((n_items, self.n_edges), dtype=torch.long, device=item_item_sim.device)

        for i in range(n_items):
            sim_scores = item_item_sim[i]
            # Don't include self
            sim_scores[i] = -float('inf')
            # Get top-k neighbors
            _, neighbors = torch.topk(sim_scores, self.n_edges, dim=0)
            adjacency[i] = neighbors

        return adjacency

    def init_graph(self):
        """Initialize graph for constrained decoding."""
        self.log("[MODEL] Building item-item similarity matrix...")
        item_item_sim = self.build_ii_sim_mat()
        
        self.log("[MODEL] Building adjacency list...")
        adjacency = self.build_adjacency_list(item_item_sim)
        
        self.decoding_graph = adjacency
        self.init_flag = True

    def log(self, message: str, level: str = 'info'):
        """Log messages using the configured logger."""
        from genrec.utils import log
        return log(message, self.config['accelerator'], self.logger, level=level)

    def graph_propagation(self, token_logits, n_return_sequences):
        """
        Graph-based search to find valid items constrained by the k-NN graph.
        
        This is a simplified version that uses random initialization and graph traversal
        to find semantically similar items during decoding.
        
        Args:
            token_logits: (batch_size, vocab_size) model predictions
            n_return_sequences: Number of items to recommend
        
        Returns:
            predictions: (batch_size, n_return_sequences, 1) top item indices
            visited_counts: (batch_size, 1) number of items explored
        """
        batch_size = token_logits.shape[0]
        visited_nodes = {}
        for batch_id in range(batch_size):
            visited_nodes[batch_id] = set()

        # Random initialization: start with num_beams random items
        topk_nodes_sorted = torch.randint(
            1, self.dataset.n_items,
            (batch_size, self.num_beams),
            dtype=torch.long,
            device=token_logits.device
        )

        # Track initial items
        for batch_id in range(batch_size):
            for node in topk_nodes_sorted[batch_id].cpu().numpy().tolist():
                visited_nodes[batch_id].add(node)

        # Iterative graph traversal
        for _ in range(self.propagation_steps):
            all_neighbors = self.adjacency[topk_nodes_sorted].view(batch_size, -1)
            next_nodes = []
            
            for batch_id in range(batch_size):
                neighbors_in_batch = torch.unique(all_neighbors[batch_id])
                
                for node in neighbors_in_batch.cpu().numpy().tolist():
                    visited_nodes[batch_id].add(node)
                
                # Rank neighbors by model confidence
                neighbor_tokens = self.item_id2tokens[neighbors_in_batch]  # (n_neighbors, n_digit)
                neighbor_logits = token_logits[batch_id][neighbor_tokens - 1].mean(dim=-1)  # (n_neighbors,)
                
                # Select top num_beams
                k = min(self.num_beams, neighbor_logits.shape[0])
                _, top_indices = torch.topk(neighbor_logits, k)
                next_nodes.append(neighbors_in_batch[top_indices])
            
            # Pad shorter sequences to maintain batch shape
            max_len = max(node.shape[0] for node in next_nodes)
            padded_nodes = []
            for node in next_nodes:
                if node.shape[0] < max_len:
                    padded = torch.cat([node, torch.zeros(max_len - node.shape[0], dtype=node.dtype, device=node.device)])
                else:
                    padded = node[:max_len]
                padded_nodes.append(padded)
            
            topk_nodes_sorted = torch.stack(padded_nodes, dim=0)[:, :self.num_beams]

        visited_counts = torch.FloatTensor([[len(visited_nodes[batch_id])] for batch_id in range(batch_size)])
        return topk_nodes_sorted[:, :n_return_sequences].unsqueeze(-1), visited_counts

    def generate(self, batch, n_return_sequences=1):
        """
        Generate next items given a user's history.
        
        Supports both direct greedy decoding and graph-constrained beam search.
        
        Args:
            batch: Dictionary with 'input_ids', 'seq_lens', 'attention_mask'
            n_return_sequences: Number of items to recommend (k in top-k)
        
        Returns:
            predictions: (batch_size, n_return_sequences, 1) recommended item indices
            OR
            (predictions, visited_counts) if using graph-constrained decoding
        """
        # Forward pass without loss
        outputs = self.forward(batch, return_loss=False)
        
        # Extract prediction at sequence end
        states = outputs.final_states.gather(
            dim=1,
            index=(batch['seq_lens'] - 1).view(-1, 1, 1, 1).expand(-1, 1, self.n_pred_head, self.config['n_embd'])
        )
        
        # Normalize and compute token logits
        states = F.normalize(states, dim=-1)
        token_emb = self.gpt2.wte.weight[1:-1]
        token_emb = F.normalize(token_emb, dim=-1)
        token_embs = torch.chunk(token_emb, self.n_pred_head, dim=0)
        
        logits = [torch.matmul(states[:, 0, i, :], token_embs[i].T) / self.temperature 
                 for i in range(self.n_pred_head)]
        logits = [F.log_softmax(logit, dim=-1) for logit in logits]
        token_logits = torch.cat(logits, dim=-1)
        
        # Decode items
        if self.generate_w_decoding_graph:
            if not self.init_flag:
                self.init_graph()
                self.init_flag = True
            
            return self.graph_propagation(
                token_logits=token_logits,
                n_return_sequences=n_return_sequences
            )
        else:
            # Direct greedy decoding
            item_logits = torch.gather(
                input=token_logits.unsqueeze(-2).expand(-1, self.dataset.n_items, -1),
                dim=-1,
                index=(self.item_id2tokens[1:, :] - 1).unsqueeze(0).expand(token_logits.shape[0], -1, -1)
            ).mean(dim=-1)
            
            preds = item_logits.topk(n_return_sequences, dim=-1).indices + 1
            return preds.unsqueeze(-1)