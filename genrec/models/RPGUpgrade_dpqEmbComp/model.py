import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model
from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer

class DPQ(nn.Module):
    # Differentiable Product Quantization.
    def __init__(self, d: int, D: int, n_clusters: int, v_dim: int, tokenizer):
        super().__init__()
        assert d % D == 0, f"Sentence embedding dim ({d}) must be divisible by D ({D})"
        print(f'[MODEL] Initializing DPQ module...')
        self.d = d
        self.D = D
        self.n_clusters = n_clusters
        self.sub_dim = d // D
        self.v_dim = v_dim
        
        # --- Learnable linear projection R -----------------------------------
        # Unconstrained nn.Linear(d, d, bias=False): y = x @ weight^T.
        # Warm-initialised from the FAISS OPQ transform so that R and K start
        # in a consistent state.  No orthogonality constraint is imposed —
        # the model is free to learn any linear transformation.
        self.rotation = nn.Linear(d, d, bias=False)
        if tokenizer.opq_rotation is not None:
            print(f'[MODEL] Warm-initializing rotation from FAISS OPQ transform')
            with torch.no_grad():
                self.rotation.weight.copy_(
                    torch.from_numpy(tokenizer.opq_rotation)
                )
        else:
            print(f'[MODEL] opq_rotation unavailable — rotation uses default init')

        # --- Key codebooks K (D, n_clusters, sub_dim) ------------------------
        print('[MODEL] Creating K matrix')
        if tokenizer.pq_codebooks is not None:
            print(f'[MODEL] Using pre-trained PQ codebooks...')
            K_init = torch.from_numpy(tokenizer.pq_codebooks).float()  # (D, K, sub_dim)
        else:
            print(f'[MODEL] Initializing random PQ codebooks...')
            K_init = torch.randn(D, n_clusters, self.sub_dim) * 0.02
        self.K = nn.Parameter(K_init)

        # --- Value codebooks V (D, n_clusters, v_dim) ------------------------
        print('[MODEL] Creating V matrix')
        if v_dim == self.sub_dim and tokenizer.pq_codebooks is not None:
            print(f'[MODEL] Using pre-trained PQ value codebooks...')
            V_init = torch.from_numpy(tokenizer.pq_codebooks).float()
        else:
            print(f'[MODEL] Initializing random PQ value codebooks...')
            V_init = torch.randn(D, n_clusters, v_dim) * 0.02
        self.V = nn.Parameter(V_init)

    def forward(self, x: torch.Tensor, tau: float = 1.0, sigma: float = 1.0) -> dict:
        
        # Notation used below:
        #     B : batch size
        #     L : sequence length
        #     d : full sentence embedding dim
        #     D : number of PQ subspaces
        #     K : number of clusters per subspace (n_clusters)
        #     v : value dimension per subspace (v_dim)

        # Args:  x   : (B, L, d) sentence embeddings
        B, L, _ = x.shape

        # 1) Rotate from original sentence-embedding space to PQ-aligned space.
        x_rot = self.rotation(x)                              # (B, L, d)

        # 2) Split each d-dim vector into D independent sub-vectors.
        x_sub = x_rot.view(B, L, self.D, self.sub_dim)       # (B, L, D, sub_dim)

        # 3) Compute assignment logits to each cluster center per subspace.
        logits = torch.einsum('bldi,dki->bldk', x_sub, self.K)  # (B, L, D, n_clusters)

        # 4 & 5) Turn logits into soft probabilities AND hard assignments
        if self.training:
            raw_gumbel = -torch.log(-torch.log(torch.rand_like(logits).clamp(min=1e-10)) + 1e-10)
            
            # Scale the Gumbel noise by the uncertainty standard deviation (SDUD)
            noisy_logits = logits + (sigma * raw_gumbel)
            
            soft_probs = F.softmax(noisy_logits / tau, dim=-1)
            codes = noisy_logits.argmax(dim=-1)
        else:
            soft_probs = F.softmax(logits / tau, dim=-1)
            codes = logits.argmax(dim=-1)
        
        # 6) Hard reconstruction by gathering the selected row from value codebook V.
        # Shift codes by their subspace offsets (0, K, 2K...)
        offsets = torch.arange(self.D, device=codes.device) * self.n_clusters
        flat_codes = codes + offsets # (B, L, D)
        # Flatten the V matrix and perform direct dictionary lookup
        flat_V = self.V.view(self.D * self.n_clusters, self.v_dim)
        hard = F.embedding(flat_codes, flat_V) # (B, L, D, v_dim)
        
        # 7) Soft reconstruction using weighted average of all K codewords.
        soft = torch.einsum('bldk,dkv->bldv', soft_probs, self.V)  # (B, L, D, v_dim)

        # 8) Straight-Through Estimator:
        ste = hard + soft - soft.detach()                     # (B, L, D, v_dim)

        # 9) Mean over the D dimension to produce GPT-facing token embeddings.
        return {
            'ste':   ste.mean(dim=-2),
            'soft':  soft.mean(dim=-2),
            'hard':  hard.mean(dim=-2),
            'codes': codes,
        }

class ResBlock(nn.Module):
    # Residual block: x + SiLU(Linear(x)).  Initialized as identity.

    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.act(self.linear(x))

class RPGUpgrade_dpqEmbComp(AbstractModel):
    # GPT-2 Sequential Recommendation with end-to-end Differentiable PQ.

    # Compared to RPG the only change is in how input embeddings are produced:
    #     RPG       : item_ids → item2tokens (frozen) → GPT-2 wte → mean-pool → GPT-2
    #     RPGUpgrade: item_ids → frozen sent_emb_table → DPQ (learnable) → proj → GPT-2    
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer,
    ):
        super().__init__(config, dataset, tokenizer)
        print(f'[MODEL] Initializing RPGUpgrade model...')
        FREEZE_SENT_EMB = True  
        sent_embs_tensor = torch.from_numpy(tokenizer.sent_embs)   # (n_items, d)
        self.sent_emb_table = nn.Embedding.from_pretrained(
            sent_embs_tensor, freeze=FREEZE_SENT_EMB, padding_idx=0
        )
        self.sent_emb_dim: int = sent_embs_tensor.shape[1]         # d

        # ------------------------------------------------------------------
        # Differentiable PQ module
        # ------------------------------------------------------------------
        v_dim = config.get('dpq_v_dim', config['n_embd'] // config['n_codebook'])
        print(f"v_dim is calculated by dpq_v_dim config or n_embd // n_codebook: {v_dim}")
        self.dpq = DPQ(
            d=self.sent_emb_dim,
            D=config['n_codebook'],
            n_clusters=config['codebook_size'],
            v_dim=v_dim,
            tokenizer=tokenizer,
        )
        dpq_out_dim = v_dim 

        # Optional projection if DPQ output dim ≠ GPT-2 hidden dim
        if dpq_out_dim != config['n_embd']:
            self.input_proj: nn.Module = nn.Linear(dpq_out_dim, config['n_embd'])
        else:
            self.input_proj = nn.Identity()

        if dpq_out_dim != config['n_embd']:
            self.output_proj: nn.Module = nn.Linear(config['n_embd'], dpq_out_dim)
        else:
            self.output_proj = nn.Identity()

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

        # Gumbel temperature 
        self.gumbel_tau: float = config.get('quantizer_temperature', 1.0)
        self.gumbel_tau_min: float = config.get('min_quantizer_temperature', 0.1)
        self.gumbel_tau_decay: float = config.get('quantizer_temperature_decay', 0.9)
        
         # SDUD Parameters
        self.sigma = 1.0  # Initial noise scale
        self.lambda_val = 1.2  # The paper recommends tuning between 1.0 and 2.0

    def anneal_tau(self):
        # Exponential decay with floor: tau <- max(tau_min, tau * tau_decay).
        self.gumbel_tau = max(
            self.gumbel_tau_min,
            self.gumbel_tau * self.gumbel_tau_decay,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _map_item_tokens(self) -> torch.Tensor:
        # Create lookup table: item_id → 32-digit semantic code (same as RPG).
        item_id2tokens = torch.zeros(
            (self.dataset.n_items, self.tokenizer.n_digit), dtype=torch.long
        )
        for item in self.tokenizer.item2tokens:
            item_id = self.dataset.item2id[item]
            item_id2tokens[item_id] = torch.LongTensor(self.tokenizer.item2tokens[item])
        return item_id2tokens

    def _get_all_item_embs(self) -> torch.Tensor:
        
        # Return normalized DPQ hard-reconstructed embeddings for all real items.

        # Returns:
        #     item_embs: (n_items - 1, dpq_out_dim), item id 1..n_items-1 maps to row 0..n_items-2.
        
        all_sent = self.sent_emb_table.weight[1:].unsqueeze(0)           # (1, n_items-1, d)
        item_embs = self.dpq(all_sent, tau=self.gumbel_tau)['hard']      # (1, n_items-1, dpq_out_dim)
        return F.normalize(item_embs.squeeze(0), dim=-1)                 # (n_items-1, dpq_out_dim)

    def build_ii_sim_mat(self) -> torch.Tensor:
        
        # Build item-item similarity matrix from DPQ hard embeddings.

        # Similarity is cosine in DPQ space (dot product after L2 normalization),
        # then mapped from [-1, 1] to [0, 1] for consistency with RPG graph init.
        
        n_items = self.dataset.n_items
        item_embs = self._get_all_item_embs()  # (n_items-1, dpq_out_dim), ids 1.. map to rows 0..
        item_item_sim = torch.zeros(
            (n_items, n_items), device=item_embs.device, dtype=torch.float32
        )

        for i_start in range(1, n_items, self.chunk_size):
            i_end = min(i_start + self.chunk_size, n_items)
            emb_i = item_embs[i_start - 1:i_end - 1]  # (bi, dpq_out_dim)

            for j_start in range(1, n_items, self.chunk_size):
                j_end = min(j_start + self.chunk_size, n_items)
                emb_j = item_embs[j_start - 1:j_end - 1]  # (bj, dpq_out_dim)

                sims = emb_i @ emb_j.T                   # (bi, bj), cosine due to normalization
                sims_01 = 0.5 * (sims + 1.0)             # map [-1, 1] -> [0, 1]
                item_item_sim[i_start:i_end, j_start:j_end] = sims_01

        return item_item_sim

    def build_adjacency_list(self, item_item_sim: torch.Tensor) -> torch.Tensor:
        # Find top-k nearest neighbors for each item.
        return torch.topk(item_item_sim, k=self.n_edges, dim=-1).indices

    def init_graph(self):
        # Initialize k-NN graph for graph-constrained decoding.
        self.tokenizer.log("Building item-item similarity matrix...")
        item_item_sim = self.build_ii_sim_mat()
        self.adjacency = self.build_adjacency_list(item_item_sim)
        self.tokenizer.log("Graph initialized.")

    def graph_propagation(self, token_logits: torch.Tensor, n_return_sequences: int):
        # Graph-based search constrained by k-NN graph.
        batch_size = token_logits.shape[0]
        visited_nodes = {}
        for batch_id in range(batch_size):
            visited_nodes[batch_id] = set()

        # Random initialization
        topk_nodes_sorted = torch.randint(
            1, self.dataset.n_items, (batch_size, self.num_beams), 
            dtype=torch.long, device=token_logits.device
        )
        
        # Track initial items as visited
        for batch_id in range(batch_size):
            for node in topk_nodes_sorted[batch_id].cpu().numpy().tolist():
                visited_nodes[batch_id].add(node)
                
        # Iterative graph traversal
        for sid in range(self.propagation_steps):
            all_neighbors = self.adjacency[topk_nodes_sorted].view(batch_size, -1)
            next_nodes = []
            
            for batch_id in range(batch_size):
                neighbors_in_batch = torch.unique(all_neighbors[batch_id])
                for node in neighbors_in_batch.cpu().numpy().tolist():
                    visited_nodes[batch_id].add(node)
                    
                # Score neighbors using the parallel token logits!
                scores = torch.gather(
                    input=token_logits[batch_id].unsqueeze(0).expand(neighbors_in_batch.shape[0], -1),
                    dim=-1,
                    index=(self.item_id2tokens[neighbors_in_batch] - 1)
                ).mean(dim=-1)
                
                # Select top candidates
                idxs = torch.topk(scores, self.num_beams).indices
                next_nodes.append(neighbors_in_batch[idxs])
                
            topk_nodes_sorted = torch.stack(next_nodes, dim=0)
            
        visited_counts = torch.FloatTensor([[len(visited_nodes[batch_id])] for batch_id in range(batch_size)])
        return topk_nodes_sorted[:, :n_return_sequences].unsqueeze(-1), visited_counts

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


    def forward(self, batch: dict, return_loss: bool = True):
        
        # Symbols:
        #     B : batch size
        #     L : sequence length
        #     d : sentence embedding dimension
        #     E : GPT hidden dimension (config['n_embd'])
        #     H : number of prediction heads (= tokenizer.n_digit)
        #     M : number of valid training positions in this batch (label != -100)
        #     I : number of real items (= n_items - 1, excluding padding id 0)

        # 1. Look up frozen sentence embeddings for each item in the sequence.
        # 2. Pass through DPQ → STE output feeds into GPT-2.
        # 3. Apply prediction heads.
        # 4. Optionally compute cross-entropy loss over labeled positions.
        

        # 1) Item id lookup.
        #    batch['input_ids']: (B, L) integer item ids
        #    sent_embs         : (B, L, d)
        sent_embs = self.sent_emb_table(batch['input_ids'])    # (B, L, d)

        # 2) Differentiable quantization + optional dimension projection.
        #    dpq_out['ste']: (B, L, dpq_out_dim)
        #    input_embs    : (B, L, E) after self.input_proj
        dpq_out = self.dpq(sent_embs, tau=self.gumbel_tau, sigma=self.sigma)     
        input_embs = self.input_proj(dpq_out['ste'])            # (B, L, n_embd)

        # 3) GPT-2 contextual encoding.
        #    inputs_embeds          : (B, L, E)
        #    attention_mask         : (B, L) (1=keep, 0=masked)
        #    outputs.last_hidden_state: (B, L, E)
        outputs = self.gpt2(
            inputs_embeds=input_embs,
            attention_mask=batch['attention_mask'],
        )

        # 4) Run H residual prediction heads in parallel over GPT hidden states.
        #    per-head output: (B, L, E)
        #    stack along head axis -> final_states: (B, L, H, E)
        final_states = torch.cat(
            [self.pred_heads[i](outputs.last_hidden_state).unsqueeze(-2)
             for i in range(self.n_pred_head)],
            dim=-2,
        )                                                       # (B, L, D, n_e mbd)
        outputs.final_states = final_states

        # 5. Loss
        if return_loss:
            assert 'labels' in batch, 'Batch must contain labels.'
            # Mask to filter out padding (-100)
            label_mask = batch['labels'].view(-1) != -100
            
            # Get the ground truth item IDs and fetch their continuous embeddings
            target_ids = batch['labels'].view(-1)[label_mask]
            target_sent_embs = self.sent_emb_table(target_ids) # (N_valid d)

            # Pass targets through DPQ to dynamically get the discrete ground-truth codes
            target_codes = self.dpq(target_sent_embs.unsqueeze(1), tau=self.gumbel_tau, sigma=self.sigma)['codes'].squeeze(1) 
        
            # Extract prediction states and filter valid positions
            # final_states is (B, L, 32, E) -> selected_states is (N_valid, 32, E)
            selected_states = final_states.view(-1, self.n_pred_head, self.config['n_embd'])[label_mask]
            selected_states = F.normalize(selected_states, dim=-1)
            
            # Access the V matrix! Normalize it for cosine similarity scoring
            # V_norm shape: (32, 256, 448)
            V_norm = F.normalize(self.dpq.V, dim=-1)

            # Compute logits and cross-entropy loss for each of the 32 digits
            losses = []
            for i in range(self.n_pred_head):
                # Dot product the i-th head's output with the i-th V codebook
                # (N_valid, 448) @ (448, 256) -> (N_valid, 256)
                token_logits = torch.matmul(selected_states[:, i, :], V_norm[i].T) / self.temperature
                
                # Calculate loss against the dynamically generated target codes
                loss_i = self.loss_fct(token_logits, target_codes[:, i])
                losses.append(loss_i)
                
            outputs.loss = torch.mean(torch.stack(losses))
            
            # --- Uncertainty Decay (SDUD) ---
            # Automatically scale the noise for the next batch based on current loss!
            if self.training:
                current_loss = outputs.loss.detach()
                self.sigma = max(0.0, torch.sqrt(current_loss).item() - self.lambda_val)
            
        return outputs

    def generate(self, batch: dict, n_return_sequences: int = 1, return_loss: bool = False):
        # Forward pass
        outputs = self.forward(batch, return_loss=return_loss)
        
        # Extract last state for each sequence in the batch
        states = outputs.final_states.gather(
            dim=1, index=(batch['seq_lens'] - 1).view(-1, 1, 1, 1).expand(
                -1, 1, self.n_pred_head, self.config['n_embd']
            ),
        ) # Shape: (B, 1, 32, 448)
        states = F.normalize(states, dim=-1)

        # ---------------------------------------------------------
        # Dynamically update the semantic IDs before decoding
        # ---------------------------------------------------------
        with torch.no_grad():
            all_sent = self.sent_emb_table.weight # (n_items, d)
            all_codes = self.dpq(all_sent.unsqueeze(0), tau=self.gumbel_tau)['codes'].squeeze(0) # (n_items, 32)
            # Add offset for concatenated token_logits indexing (0, 256, 512...)
            offsets = torch.arange(self.n_pred_head, device=states.device) * self.dpq.n_clusters
            # Update the global map (adding +1 to maintain 1-based indexing)
            self.item_id2tokens = all_codes + offsets + 1 
        
        # ---------------------------------------------------------
        # Generate parallel token logits
        # ---------------------------------------------------------
        V_norm = F.normalize(self.dpq.V, dim=-1)
        logits = []
        for i in range(self.n_pred_head):
            logit = torch.matmul(states[:, 0, i, :], V_norm[i].T) / self.temperature
            logits.append(F.log_softmax(logit, dim=-1))
        
        token_logits = torch.cat(logits, dim=-1) # Shape: (B, 32 * 256)

        # ---------------------------------------------------------
        # Decode items (Graph search or Direct Top-K)
        # ---------------------------------------------------------
        if self.generate_w_decoding_graph:
            if not self.init_flag:
                self.init_graph()
                self.init_flag = True
            
            # The graph will now correctly use the updated self.item_id2tokens
            preds = self.graph_propagation(
                token_logits=token_logits, 
                n_return_sequences=n_return_sequences,
            )
        else:
            # Direct greedy decoding against all items
            item_logits = torch.gather(
                input=token_logits.unsqueeze(-2).expand(-1, self.dataset.n_items, -1),
                dim=-1,
                index=(self.item_id2tokens[1:,:] - 1).unsqueeze(0).expand(token_logits.shape[0], -1, -1)
            ).mean(dim=-1)
            
            preds = item_logits.topk(n_return_sequences, dim=-1).indices + 1
            preds = preds.unsqueeze(-1)
        
        if return_loss:
            return preds, outputs.loss
        return preds
