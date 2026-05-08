# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
RPG (Residual Quantization Product) Model for Sequential Recommendation

This module implements a GPT-2 based recommendation model that:
1. Represents each item as a product of 32 quantized codes (32-digit semantic ID)
2. Uses GPT-2 to model sequential dependencies in user purchase history
3. Predicts the next item by predicting all 32 digits in parallel
4. Optionally uses graph-constrained decoding to ensure predictions are semantically valid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model

from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer


class ResBlock(nn.Module):
    """
    A lightweight Residual Block for refining predictions.
    
    Architecture: x + SiLU(Linear(x))
    - Initialized as identity (zero weights) so it doesn't disrupt initial predictions
    - SiLU activation adds non-linearity for learning
    
    Input shape:  (batch_size, seq_len, hidden_size)
    Output shape: (batch_size, seq_len, hidden_size)
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        # Initialize as identity mapping (linear.weight = 0, linear.bias = 0)
        torch.nn.init.zeros_(self.linear.weight)
        # SiLU activation (Swish): x * sigmoid(x)
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Forward pass: apply residual connection with learned non-linearity.
        
        Args:
            x: Input tensor of shape (..., hidden_size)
        
        Returns:
            x + SiLU(Linear(x)) - preserves information while learning refinements
        """
        return x + self.act(self.linear(x))


class RPG(AbstractModel):
    """
    RPG: A GPT-2 based Sequential Recommendation Model with Product Quantization
    
    Architecture Overview:
    ├─ Tokenizer: Encodes items as 32-digit semantic codes (product quantization)
    ├─ Item→Token Lookup: Maps item_id → 32 digit tokens
    ├─ GPT-2 Backbone: Models sequential patterns in user interactions
    ├─ Prediction Heads: 32 parallel ResBlocks, each predicting one digit
    ├─ Loss: 32 separate classification losses (one per digit)
    └─ Inference: Reconstruct items from predicted digits, optionally constrain with graph
    
    Tensor shape conventions throughout:
    - batch_size: typically 4-32
    - seq_len: typically 5-100 (history length)
    - n_digit: 32 (fixed, for product quantization)
    - codebook_size: 256 (fixed, per digit)
    - n_embd: 768 (GPT-2 hidden dimension)
    """
    
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer
    ):
        """
        Initialize the RPG model.
        
        Args:
            config: Configuration dictionary with keys like 'n_embd', 'n_layer', etc.
            dataset: Dataset with n_items, item2id mappings
            tokenizer: Tokenizer with item2tokens mapping and vocab info
        """
        super(RPG, self).__init__(config, dataset, tokenizer)

        # ============ STEP 1: Create item_id → token mapping ============
        # Maps each item_id to its 32-digit semantic code
        # Shape: (n_items, 32)
        self.item_id2tokens = self._map_item_tokens().to(self.config['device'])

        # ============ STEP 2: Initialize GPT-2 backbone ============
        gpt2config = GPT2Config(
            vocab_size=tokenizer.vocab_size,        # 8193 (32 digits × 256 + special tokens)
            n_positions=tokenizer.max_token_seq_len, # From config.yaml: max_item_seq_len=50
            n_embd=config['n_embd'],                 # From config.yaml: n_embd=448 (embedding/hidden dim)
            n_layer=config['n_layer'],               # From config.yaml: n_layer=2 (transformer blocks)
            n_head=config['n_head'],                 # From config.yaml: n_head=4 (attention heads)
            n_inner=config['n_inner'],               # From config.yaml: n_inner=1024 (FFN intermediate)
            activation_function=config['activation_function'],
            resid_pdrop=config['resid_pdrop'],       # residual dropout
            embd_pdrop=config['embd_pdrop'],         # embedding dropout
            attn_pdrop=config['attn_pdrop'],         # attention dropout
            layer_norm_epsilon=config['layer_norm_epsilon'],
            initializer_range=config['initializer_range'],
            eos_token_id=tokenizer.eos_token,
        )
        # GPT-2 Token Embedding matrix shape: (vocab_size=8193, n_embd=768)
        self.gpt2 = GPT2Model(gpt2config)

        # ============ STEP 3: Create 32 parallel prediction heads ============
        # One ResBlock per digit to refine the GPT-2 hidden states
        # Input: (batch, seq_len, 768) → Output: (batch, seq_len, 768)
        self.n_pred_head = self.tokenizer.n_digit  # 32
        pred_head_list = []
        for i in range(self.n_pred_head):
            pred_head_list.append(ResBlock(self.config['n_embd']))
        self.pred_heads = nn.Sequential(*pred_head_list)

        # ============ STEP 4: Setup loss and generation parameters ============
        # From config.yaml: temperature=0.07 (lower temp = sharper predictions)
        self.temperature = self.config['temperature']
        # Ignore padding tokens (-100) when computing loss
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.ignored_label)

        # Graph-constrained decoding parameters
        self.generate_w_decoding_graph = False      # flag to enable/disable graph search at inference
        self.init_flag = False                       # has graph been initialized?
        # From config.yaml: chunk_size=1024 (process similarity in chunks for memory efficiency)
        self.chunk_size = config['chunk_size']
        # From config.yaml: num_beams=50 (beam width for graph search)
        self.num_beams = config['num_beams']
        # From config.yaml: n_edges=50 (top-k nearest neighbors per item)
        self.n_edges = config['n_edges']
        # From config.yaml: propagation_steps=3 (iterations of graph traversal)
        self.propagation_steps = config['propagation_steps']

    def _map_item_tokens(self) -> torch.Tensor:
        """
        Create a lookup table: item_id → 32-digit semantic code.
        
        Example with n_items=100:
        - item "iPhone_13" has item_id=5
        - tokenizer.item2tokens["iPhone_13"] = (23, 145, 67, ..., 200)  [32 values]
        - This function creates: item_id2tokens[5] = (23, 145, 67, ..., 200)
        
        Returns:
            torch.Tensor of shape (n_items, 32) where each row is the semantic code
        """
        item_id2tokens = torch.zeros((self.dataset.n_items, self.tokenizer.n_digit), dtype=torch.long)
        for item in self.tokenizer.item2tokens:
            item_id = self.dataset.item2id[item]
            # Store the 32-digit code for this item
            item_id2tokens[item_id] = torch.LongTensor(self.tokenizer.item2tokens[item])
        return item_id2tokens

    @property
    def n_parameters(self) -> str:
        """
        Calculate and format the number of trainable parameters.
        
        Returns:
            str: Breakdown of embedding params, non-embedding params, and total
        """
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        emb_params = sum(p.numel() for p in self.gpt2.get_input_embeddings().parameters() if p.requires_grad)
        return f'#Embedding parameters: {emb_params}\n' \
                f'#Non-embedding parameters: {total_params - emb_params}\n' \
                f'#Total trainable parameters: {total_params}\n'

    def forward(self, batch: dict, return_loss=True) -> torch.Tensor:
        """
        Forward pass: Item sequence → predict next item's digit codes.
        
        TENSOR SHAPE WALKTHROUGH (example: batch_size=4, seq_len=10):
        
        INPUT:
          batch['input_ids']:      (4, 10)     - item indices from dataset
          batch['attention_mask']: (4, 10)     - 1 for valid tokens, 0 for padding
          batch['labels']:         (4, 10)     - next item indices (only last column used in training)
        
        STEP 1 - Token Lookup:
          item_id2tokens lookup: (4, 10) → (4, 10, 32)  
          Now each item is represented by 32 semantic codes
        
        STEP 2 - Token Embedding:
          GPT-2 embedding layer: (4, 10, 32) → (4, 10, 32, 768)
          Average across 32 codes: (4, 10, 32, 768) → (4, 10, 768)
          Now: compact representation of each item
        
        STEP 3 - GPT-2 Encoding:
          Process through 12 transformer layers with self-attention
          Input:  (4, 10, 768)
          Output: (4, 10, 768)  - contextualized representations
          Each position now captures dependencies on previous items
        
        STEP 4 - Prediction Heads (32 parallel ResBlocks):
          Apply 32 independent ResBlocks to refine predictions
          Each: (4, 10, 768) → (4, 10, 768)
          Stack them: (4, 10, 32, 768)
          Now: 32 separate prediction spaces for 32 digits
        
        STEP 5 - Loss Computation (if return_loss=True):
          Extract valid positions (ignore padding): ~40 tokens (4×10 minus padding)
          Normalize predictions: L2 normalize to unit sphere
          Split by digit: 32 chunks of shape (40, 1, 768)
          Compute logits: matrix multiply with embedding weights (40, 256) per digit
          Compute loss: cross-entropy for each digit separately
          Output: single scalar loss (mean of 32 losses)
        
        Args:
            batch: Dict with 'input_ids', 'attention_mask', 'labels', 'seq_lens'
            return_loss: If True, compute training loss; if False, only return logits
        
        Returns:
            outputs: Object with .loss (if return_loss=True) and .final_states
        """
        
        # ============ STEP 1: Token Lookup ============
        # Convert item_ids to their semantic digit codes
        # Input:  batch['input_ids'] shape (batch_size, seq_len)
        # Lookup: self.item_id2tokens shape (n_items, 32)
        # Output: (batch_size, seq_len, 32)
        input_tokens = self.item_id2tokens[batch['input_ids']]
        
        # ============ STEP 2: Token Embedding & Averaging ============
        # Embed each digit code into a 448-dimensional vector (from config.yaml: n_embd=448)
        # self.gpt2.wte: embedding table of shape (vocab_size=8193, n_embd=448)
        # input_tokens: (batch_size, seq_len, 32)
        # After embedding: (batch_size, seq_len, 32, 448)
        # After mean(dim=-2): (batch_size, seq_len, 448)
        #   ^ Average the 32 embeddings to get a single compact representation per item
        input_embs = self.gpt2.wte(input_tokens).mean(dim=-2)
        
        # ============ STEP 3: GPT-2 Transformer Encoding ============
        # Process sequence through 2 transformer layers (from config.yaml: n_layer=2)
        # Each layer has 4 attention heads (from config.yaml: n_head=4)
        # Computes multi-head attention between all positions to capture dependencies
        # Input:  input_embs (batch_size, seq_len, 448)
        #         attention_mask (batch_size, seq_len)
        # Output: outputs.last_hidden_state (batch_size, seq_len, 448)
        outputs = self.gpt2(
            inputs_embeds=input_embs,
            attention_mask=batch['attention_mask']
        )
        
        # ============ STEP 4: Apply 32 Prediction Heads ============
        # For each of 32 digit positions (from config.yaml: n_codebook=32), apply a separate ResBlock
        # Each ResBlock refines the 448-dimensional hidden state from GPT-2
        # Each ResBlock: (batch_size, seq_len, 448) → (batch_size, seq_len, 448)
        # Result: stack all 32 into a single tensor
        # After list comprehension: list of 32 tensors, each (batch_size, seq_len, 1, 448)
        # After unsqueeze(-2) and cat: (batch_size, seq_len, 32, 448)
        final_states = [self.pred_heads[i](outputs.last_hidden_state).unsqueeze(-2) 
                       for i in range(self.n_pred_head)]
        final_states = torch.cat(final_states, dim=-2)
        
        # Attach to outputs object for inference
        outputs.final_states = final_states
        
        # ============ STEP 5: Loss Computation (Training Only) ============
        if return_loss:
            assert 'labels' in batch, 'The batch must contain the labels.'
            
            # ========== SUB-STEP 5.1: Create mask for valid positions ==========
            # batch['labels']: (batch_size, seq_len) e.g., (4, 10)
            # .view(-1): (batch_size*seq_len,) e.g., (40,)
            # label_mask = (!=  -100): (40,) boolean, True where label is valid
            # Example: [T, T, T, ..., T, F, F] (last 2 are padding, marked as -100)
            label_mask = batch['labels'].view(-1) != -100
            # label_mask.sum() = num_valid positions to include in loss calculation
            
            # ========== SUB-STEP 5.2: Extract predictions at valid positions ==========
            # final_states: (batch_size, seq_len, 32, 448) e.g., (4, 10, 32, 448)
            #   - dim 0: batch_size=4 samples
            #   - dim 1: seq_len=10 positions
            #   - dim 2: n_pred_head=32 (32 digit predictions per position, config.yaml: n_codebook=32)
            #   - dim 3: n_embd=448 (embedding dimension, config.yaml: n_embd=448)
            #
            # .view(-1, 32, 448): (40, 32, 448) - flatten batch and sequence dimensions
            # [label_mask]: (num_valid, 32, 448) e.g., (38, 32, 448) if 2 padding positions
            #   - Only keep rows where label_mask[i] = True (non-padding positions)
            selected_states = final_states.view(-1, self.n_pred_head, self.config['n_embd'])[label_mask]
            # num_valid = selected_states.shape[0]  # e.g., 38 (40 - 2 padding)
            
            # ========== SUB-STEP 5.3: Normalize to unit sphere ==========
            # selected_states: (num_valid, 32, 448) → (num_valid, 32, 448) normalized
            # Normalizes last dimension (448-dim vectors) to unit length
            # This makes computation use cosine distance instead of Euclidean
            selected_states = F.normalize(selected_states, dim=-1)
            # After normalization: each of 448 dimensions scaled so ||v|| = 1
            
            # ========== SUB-STEP 5.4: Split predictions by digit ==========
            # selected_states: (num_valid, 32, 448) e.g., (38, 32, 448)
            # torch.chunk(tensor, 32, dim=1) splits along dim 1 (the 32 digits)
            # Result: LIST of 32 tensors
            #   - selected_states[0]: (num_valid, 1, 448) predictions for digit 0
            #   - selected_states[1]: (num_valid, 1, 448) predictions for digit 1
            #   - ...
            #   - selected_states[31]: (num_valid, 1, 448) predictions for digit 31
            selected_states = torch.chunk(selected_states, self.n_pred_head, dim=1)
            # Now we have 32 separate prediction spaces, one per digit
            
            # ========== SUB-STEP 5.5: Extract token embedding weights ==========
            # self.gpt2.wte.weight: (vocab_size=8193, n_embd=448)
            #   - vocab_size=8193 = 8192 codebook tokens + 1 special token
            #   - n_embd=448 (from config.yaml)
            # [1:-1]: removes special tokens (padding at 0, EOS at 8192)
            # Result: (8192, 448) - one embedding vector per codebook value
            #   - These 8192 values are organized as: 32 digits × 256 values per digit
            token_emb = self.gpt2.wte.weight[1:-1]
            # token_emb[0:256] = embeddings for digit 0 values
            # token_emb[256:512] = embeddings for digit 1 values
            # ...
            # token_emb[7936:8192] = embeddings for digit 31 values
            
            # ========== SUB-STEP 5.6: Normalize token embeddings ==========
            # token_emb: (8192, 448) → (8192, 448) normalized
            # Each of 8192 embedding vectors is normalized to unit length
            token_emb = F.normalize(token_emb, dim=-1)
            
            # ========== SUB-STEP 5.7: Split token embeddings by digit ==========
            # token_emb: (8192, 448)
            # torch.chunk(tensor, 32, dim=0) splits along dim 0 (8192 values into 32 groups)
            # Result: LIST of 32 tensors
            #   - token_embs[0]: (256, 448) embeddings for digit 0's codebook (256 possible values)
            #   - token_embs[1]: (256, 448) embeddings for digit 1's codebook
            #   - ...
            #   - token_embs[31]: (256, 448) embeddings for digit 31's codebook
            token_embs = torch.chunk(token_emb, self.n_pred_head, dim=0)
            # From config.yaml: codebook_size=256 (so each group has 256 rows)
            
            # ========== SUB-STEP 5.8: Compute logits for each digit (cosine similarity) ==========
            # This is the core computation: for each digit, match predictions to embeddings
            # For digit i:
            #   selected_states[i]: (num_valid, 1, 448) - predicted embedding for digit i
            #   token_embs[i]: (256, 448) - possible codebook values for digit i
            #
            #   .squeeze(dim=1): (num_valid, 448) - remove dim 1 (which is size 1)
            #   .T: (448, 256) - transpose embeddings
            #   matmul: (num_valid, 448) @ (448, 256) = (num_valid, 256)
            #     - For each valid position, 256 scores (one per possible digit value)
            #   / self.temperature (config.yaml: temperature=0.07):
            #     - Lower temperature = sharper (more confident) logits
            #     - Division by 0.07 makes values ~14x larger, sharpening probabilities
            #   Result: (num_valid, 256) logits for digit i
            #
            # After loop: LIST of 32 tensors
            #   - token_logits[0]: (num_valid, 256) logits for digit 0
            #   - token_logits[1]: (num_valid, 256) logits for digit 1
            #   - ...
            #   - token_logits[31]: (num_valid, 256) logits for digit 31
            token_logits = [
                torch.matmul(selected_states[i].squeeze(dim=1), token_embs[i].T) / self.temperature
                for i in range(self.n_pred_head)
            ]
            
            # ========== SUB-STEP 5.9: Extract ground-truth digit codes ==========
            # batch['labels']: (batch_size, seq_len) e.g., (4, 10) - target item IDs
            # .view(-1): (batch_size*seq_len,) e.g., (40,) - flatten to single dimension
            # [label_mask]: (num_valid,) e.g., (38,) - keep only valid positions
            #   - Only select positions where label_mask[i] = True (non-padding)
            # self.item_id2tokens[...]: (num_valid, 32) e.g., (38, 32)
            #   - For each valid target item, get its 32 digit codes
            #   - Each digit code is an integer 0-255 (but stored as token ID 1-8192)
            token_labels = self.item_id2tokens[batch['labels'].view(-1)[label_mask]]
            # token_labels[j, i] = ground truth digit code for digit i of item j
            # Example: token_labels[0, 0] might be 145 (encoded as token 146 in embeddings)
            
            # ========== SUB-STEP 5.10: Compute loss for each digit independently ==========
            # For each of 32 digits, compute cross-entropy loss separately
            # For digit i:
            #   token_logits[i]: (num_valid, 256) - model's predictions
            #     - dim 0: different valid positions (num_valid e.g. 38)
            #     - dim 1: confidence scores for each of 256 possible values
            #   token_labels[:, i]: (num_valid,) - ground truth digits
            #     - Each element is 0-255 (the correct digit value for that position)
            #   Adjustment: "- i * 256 - 1" converts from token ID back to digit value
            #     - token ID = digit_value + digit_index * 256 + 1
            #     - So: digit_value = token_ID - digit_index * 256 - 1
            #   self.loss_fct = CrossEntropyLoss(ignore_index=-100)
            #     - Compares 256-class logits against true class
            #     - Returns scalar loss value
            #
            #   losses[i]: scalar loss for digit i
            losses = [
                self.loss_fct(
                    token_logits[i],  # Shape: (num_valid, 256) logits for all values of digit i
                    token_labels[:, i] - i * self.config['codebook_size'] - 1  # Shape: (num_valid,) ground truth
                )
                for i in range(self.n_pred_head)  # Repeat for all 32 digits
            ]
            # losses is a LIST of 32 scalars
            # losses[0] = cross-entropy for digit 0
            # losses[1] = cross-entropy for digit 1
            # ...
            # losses[31] = cross-entropy for digit 31
            
            # ========== SUB-STEP 5.11: Average loss across all digits ==========
            # torch.stack(losses): LIST of 32 scalars → (32,) tensor
            # torch.mean(...): (32,) → scalar
            # Result: average loss across all 32 digits
            # This scalar is used for backpropagation to update model weights
            outputs.loss = torch.mean(torch.stack(losses))
        
        return outputs

    def build_ii_sim_mat(self):
        """
        Build item-item similarity matrix using semantic token embeddings.
        
        KEY IDEA: Items with similar digit codes should have high similarity.
        For each digit position, we compute cosine similarities in that digit's embedding space.
        Then average across all 32 digits to get final item-item similarity.
        
        COMPUTATION STEPS:
        1. Reshape token embedding weights into (32, 256, 768)
           - 32 digit positions
           - 256 possible values per digit
           - 768 dimensional embeddings
        
        2. Compute (256 × 256) cosine similarity matrix for each digit
           - token_sims shape: (32, 256, 256)
           - token_sims[k][i][j] = cosine_sim between i-th and j-th value of digit k
        
        3. For each pair of items, look up their similarity:
           - Item A has codes: (5, 128, 42, ...)
           - Item B has codes: (7, 125, 50, ...)
           - Similarity = average of: token_sims[0][5][7] + token_sims[1][128][125] + ...
        
        4. Do this efficiently in chunks to avoid OOM
        
        OUTPUT: item_item_sim shape (n_items, n_items)
                Symmetric matrix where entry [i][j] = similarity between item i and j
        """
        n_items = self.dataset.n_items
        # From config.yaml: n_codebook=32
        n_digit = self.tokenizer.n_digit
        # From config.yaml: codebook_size=256
        codebook_size = self.tokenizer.codebook_size

        # ============ STEP 1: Extract and reshape token embeddings ============
        # self.gpt2.wte.weight: (vocab_size=8193, embedding_dim=448) from config.yaml: n_embd=448
        # [1:-1] removes padding token and EOS token: (8192, 448)
        # Reshape to per-digit groups: (32, 256, 448)
        #   - 32 groups (one per digit, from config.yaml: n_codebook=32)
        #   - 256 possible values per digit (from config.yaml: codebook_size=256)
        #   - 448 embedding dimensions (from config.yaml: n_embd=448)
        token_embs = self.gpt2.wte.weight[1:-1].view(n_digit, codebook_size, -1)

        # ============ STEP 2: Normalize and compute similarity matrices ============
        # Normalize each embedding to unit length (for cosine similarity computation)
        # Shape stays: (32, 256, 448)
        token_embs = F.normalize(token_embs, dim=-1)
        
        # Batch matrix multiply: (32, 256, 448) @ (32, 448, 256) → (32, 256, 256)
        # Result: token_sims[k][i][j] = cosine similarity between:
        #   - the embedding of value i in digit k's codebook
        #   - the embedding of value j in digit k's codebook (from config.yaml: codebook_size=256)
        token_sims = torch.bmm(token_embs, token_embs.transpose(1, 2))

        # Convert from [-1, 1] (cosine similarity range) to [0, 1]
        # This makes the values easier to interpret (higher = more similar)
        token_sims_01 = 0.5 * (token_sims + 1.0)  # shape: (32, 256, 256)

        # ============ STEP 3: Initialize output matrix ============
        # Will accumulate similarities for all item pairs
        item_item_sim = torch.zeros((n_items, n_items), device=self.gpt2.device, dtype=torch.float32)

        # ============ STEP 4: Fill item-item matrix in chunks (memory efficient) ============
        # Process in chunks to avoid loading entire (n_items, n_items) matrix into memory
        # From config.yaml: chunk_size=1024 (process 1024 items at a time)
        for i_start in range(1, n_items, self.chunk_size):
            i_end = min(i_start + self.chunk_size, n_items)

            # Get digit codes for items i_start:i_end
            # item_id2tokens shape: (n_items, 32)
            # Slice shape: (chunk_size, 32)
            tokens_i = self.item_id2tokens[i_start:i_end]

            for j_start in range(1, n_items, self.chunk_size):
                j_end = min(j_start + self.chunk_size, n_items)

                # Get digit codes for items j_start:j_end
                tokens_j = self.item_id2tokens[j_start:j_end]

                # Initialize accumulator for this sub-block
                block_size_i = i_end - i_start
                block_size_j = j_end - j_start
                sum_block = torch.zeros((block_size_i, block_size_j), device=self.gpt2.device, dtype=torch.float32)

                # ============ STEP 5: Accumulate similarity across all 32 digits ============
                # For each of 32 digit positions (from config.yaml: n_codebook=32):
                #   For each item pair (i, j) in this chunk:
                #     Lookup: token_sims_01[k][tokens_i[i][k]][tokens_j[j][k]]
                #     Add to sum_block[i][j]
                for k in range(n_digit):
                    # Get digit values for items in this chunk
                    # row_inds: which value does item i use for digit k?
                    # Subtract (k * codebook_size + 1) to convert from token_id back to codebook index
                    row_inds = tokens_i[:, k] - k * codebook_size - 1
                    col_inds = tokens_j[:, k] - k * codebook_size - 1

                    # Look up similarities in the k-th digit's (256, 256) matrix
                    # token_sims_01[k] shape: (256, 256)
                    # .index_select(0, row_inds) selects rows: (chunk_i, 256)
                    # .index_select(1, col_inds) selects columns: (chunk_i, chunk_j)
                    temp = token_sims_01[k].index_select(0, row_inds)
                    temp = temp.index_select(1, col_inds)

                    # Add to running sum
                    sum_block += temp

                # ============ STEP 6: Average across all digits ============
                # Each item pair gets a score that's the average of 32 digit-similarities
                avg_block = sum_block / n_digit

                # Store in final matrix
                item_item_sim[i_start:i_end, j_start:j_end] = avg_block

        return item_item_sim

    def build_adjacency_list(self, item_item_sim):
        """
        Build k-NN graph: for each item, find its n_edges nearest neighbors.
        
        Args:
            item_item_sim: (n_items, n_items) similarity matrix
        
        Returns:
            adjacency: (n_items, n_edges) indices of nearest neighbors for each item
        """
        return torch.topk(item_item_sim, k=self.n_edges, dim=-1).indices

    def init_graph(self):
        """
        Initialize the k-NN graph for graph-constrained decoding.
        
        This is called once before inference if generate_w_decoding_graph=True.
        Builds the item-item similarity matrix and extracts the k-NN adjacency list.
        """
        self.tokenizer.log("Building item-item similarity matrix...")
        item_item_sim = self.build_ii_sim_mat()
        self.adjacency = self.build_adjacency_list(item_item_sim)
        self.tokenizer.log("Graph initialized.")

    def graph_propagation(self, token_logits, n_return_sequences):
        """
        Graph-based search to find valid items constrained by the k-NN graph.
        
        WHY THIS EXISTS:
        Problem: Raw model predictions might suggest items that are semantically invalid
                 (e.g., predicting a luxury watch when context suggests budget products)
        Solution: Use the k-NN graph to guide search within semantically similar items
        
        ALGORITHM:
        1. Start with random beam: num_beams candidate items (config.yaml: num_beams=50)
        2. For N iterations (config.yaml: propagation_steps=3):
           a. Find neighbors of current candidates using the k-NN graph (k=n_edges=50)
           b. Rank neighbors by model confidence (token_logits)
           c. Select top num_beams neighbors as next candidates
           d. Track all visited items
        3. Return: top n_return_sequences items from final candidates
        
        Example walkthrough (num_beams=3, propagation_steps=2):
          Iteration 0: candidates = [item_5, item_203, item_1847]
          Iteration 1: find neighbors → [item_8, item_12, item_201, ..., item_1903]
                       rank by score → [item_12, item_201, item_1903]
          Final: return top 1-3 items from [item_12, item_201, item_1903]
        
        Args:
            token_logits: (batch_size, vocab_size) raw model predictions
            n_return_sequences: how many items to return per batch sample
        
        Returns:
            predictions: (batch_size, n_return_sequences, 1) top-k item indices
            visited_counts: (batch_size, 1) number of items explored during search
        """
        batch_size = token_logits.shape[0]

        # Initialize visited tracking
        visited_nodes = {}
        for batch_id in range(batch_size):
            visited_nodes[batch_id] = set()

        # Random initialization: start with num_beams random items
        # These serve as the initial search seeds
        topk_nodes_sorted = torch.randint(
            1, self.dataset.n_items,
            (batch_size, self.num_beams),
            dtype=torch.long,
            device=token_logits.device
        )

        # Track initial items as visited
        for batch_id in range(batch_size):
            for node in topk_nodes_sorted[batch_id].cpu().numpy().tolist():
                visited_nodes[batch_id].add(node)

        # ============ Iterative graph traversal ============
        # From config.yaml: propagation_steps=3 (number of graph expansion iterations)
        for sid in range(self.propagation_steps):
            # Find all neighbors of current candidates
            # topk_nodes_sorted: (batch_size, num_beams) item indices
            # self.adjacency[items]: (batch_size, num_beams, n_edges)
            # all_neighbors: (batch_size, num_beams * n_edges) flattened neighbor list
            all_neighbors = self.adjacency[topk_nodes_sorted].view(batch_size, -1)

            next_nodes = []
            for batch_id in range(batch_size):
                # Get unique neighbors (remove duplicates)
                neighbors_in_batch = torch.unique(all_neighbors[batch_id])

                # Add to visited set
                for node in neighbors_in_batch.cpu().numpy().tolist():
                    visited_nodes[batch_id].add(node)

                # Score neighbors using model predictions
                # For each neighbor, compute the average confidence across its 32 digit codes
                scores = torch.gather(
                    input=token_logits[batch_id].unsqueeze(0).expand(neighbors_in_batch.shape[0], -1),
                    dim=-1,
                    index=(self.item_id2tokens[neighbors_in_batch] - 1)
                ).mean(dim=-1)

                # Select top num_beams neighbors by score
                idxs = torch.topk(scores, self.num_beams).indices
                next_nodes.append(neighbors_in_batch[idxs])
            
            topk_nodes_sorted = torch.stack(next_nodes, dim=0)

        # Convert to output format
        visited_counts = torch.FloatTensor([[len(visited_nodes[batch_id])] for batch_id in range(batch_size)])

        return topk_nodes_sorted[:,:n_return_sequences].unsqueeze(-1), visited_counts

    def generate(self, batch, n_return_sequences=1):
        """
        Inference: predict next item(s) for the given user history.
        
        TWO MODES:
        1. DIRECT MODE (generate_w_decoding_graph=False):
           - Use raw model predictions
           - Simply pick top-k items by score
           - Fast but may suggest semantically invalid items
        
        2. GRAPH-CONSTRAINED MODE (generate_w_decoding_graph=True):
           - Use k-NN graph to constrain search
           - Guide search towards semantically similar items
           - Slower but more coherent predictions
        
        TENSOR SHAPE WALKTHROUGH (example: batch_size=4):
        
        INPUT:
          batch['seq_lens']: (4,)  - actual sequence length for each sample
        
        STEP 1 - Forward Pass (inference):
          outputs.final_states: (4, max_seq_len, 32, 768)
        
        STEP 2 - Extract Last Position:
          Gather the state at the actual end of sequence (using seq_lens)
          states: (4, 1, 32, 768)
        
        STEP 3 - Compute Logits:
          Normalize states: (4, 1, 32, 768)
          Split by digit: 32 × (4, 1, 768)
          Matmul with token embeddings: 32 × (4, 256)
          Concatenate: (4, 8192)
          Apply softmax: (4, 8192)
        
        STEP 4A - DIRECT MODE:
          For each of 10000 items:
            Gather its 32 digit token logits
            Average confidence
          Result: (4, 10000) item scores
          Top-k: (4, n_return_sequences) item indices
        
        STEP 4B - GRAPH MODE:
          Use graph_propagation() → (4, n_return_sequences, 1) item indices
        
        Args:
            batch: Dict with 'input_ids', 'seq_lens', etc.
            n_return_sequences: Number of items to recommend per user (k in top-k)
        
        Returns:
            If direct mode: (batch_size, n_return_sequences, 1) indices
            If graph mode: (batch_size, n_return_sequences, 1) indices
        """
        
        # ============ STEP 1: Get model predictions ============
        # Forward pass without loss computation
        # outputs.final_states: (batch_size, max_seq_len, 32, 768)
        outputs = self.forward(batch, return_loss=False)
        
        # ============ STEP 2: Extract prediction at sequence end ============
        # seq_lens: (batch_size,) actual length of each sequence (max=50 from config.yaml: max_item_seq_len)
        # Create index to gather the last valid position
        # (batch_size, 1, 1, 1) broadcasted to (batch_size, 1, 32, 448)
        states = outputs.final_states.gather(
            dim=1,
            index=(batch['seq_lens'] - 1).view(-1, 1, 1, 1).expand(-1, 1, self.n_pred_head, self.config['n_embd'])
        )
        # states: (batch_size, 1, 32, 448) - last position only, all 32 digits (n_embd=448)
        
        # ============ STEP 3: Normalize predictions ============
        # Bring to unit sphere for cosine distance
        states = F.normalize(states, dim=-1)
        
        # ============ STEP 4: Compute token logits (32 digits) ============
        # Get normalized token embedding weights
        # From config.yaml: n_embd=448, n_codebook=32, codebook_size=256
        token_emb = self.gpt2.wte.weight[1:-1]  # (8192, 448)
        token_emb = F.normalize(token_emb, dim=-1)
        
        # Split into 32 groups: 32 × (256, 448)
        token_embs = torch.chunk(token_emb, self.n_pred_head, dim=0)
        
        # For each digit, compute cosine similarity with embeddings
        # From config.yaml: temperature=0.07 (lower = sharper predictions)
        # states[:, 0, i, :]: (batch_size, 448)
        # token_embs[i].T: (448, 256)
        # matmul: (batch_size, 256) - confidence for each of 256 codebook values
        # Apply temperature scaling (0.07) and softmax
        logits = [torch.matmul(states[:,0,i,:], token_embs[i].T) / self.temperature 
                 for i in range(self.n_pred_head)]
        logits = [F.log_softmax(logit, dim=-1) for logit in logits]
        
        # Concatenate all digit logits
        # From config.yaml: n_codebook=32, codebook_size=256
        # List of 32 × (batch_size, 256) → (batch_size, 8192)
        token_logits = torch.cat(logits, dim=-1)
        
        # ============ STEP 5: Decode items ============
        if self.generate_w_decoding_graph:
            # ===== GRAPH-CONSTRAINED DECODING =====
            # Use k-NN graph guided search (config.yaml: num_beams=50, n_edges=50, propagation_steps=3)
            # Guarantees recommendations are semantically similar items
            if not self.init_flag:
                self.init_graph()
                self.init_flag = True
            
            outputs = self.graph_propagation(
                token_logits=token_logits,
                n_return_sequences=n_return_sequences
            )
            return outputs
        else:
            # ===== DIRECT GREEDY DECODING =====
            # Fast inference mode: for each item, score by averaging confidence of 32 digits
            # From config.yaml: n_codebook=32
            
            # item_id2tokens[1:]: all items except padding (9999, 32)
            # Shape after gather:
            #   input: (batch_size, 8192)
            #   index: (batch_size, 9999, 32) where each [i,j,:] = digit tokens of item j
            #   output: (batch_size, 9999, 32) logits of item j's digits
            item_logits = torch.gather(
                input=token_logits.unsqueeze(-2).expand(-1, self.dataset.n_items, -1),  # (batch_size, n_items, 8192)
                dim=-1,
                index=(self.item_id2tokens[1:,:] - 1).unsqueeze(0).expand(token_logits.shape[0], -1, -1)  # (batch_size, n_items, 32)
            ).mean(dim=-1)  # → (batch_size, n_items) average across digits
            
            # Select top-k items by average confidence
            # topk returns: (values, indices) where indices shape = (batch_size, k)
            preds = item_logits.topk(n_return_sequences, dim=-1).indices + 1  # +1 to skip padding item
            
            # Add trailing dimension for consistency with graph mode
            # (batch_size, n_return_sequences) → (batch_size, n_return_sequences, 1)
            return preds.unsqueeze(-1)
