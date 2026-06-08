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

    def forward(self, x: torch.Tensor, tau: float = 1.0) -> dict:
        
        # Notation used below:
        #     B : batch size
        #     L : sequence length
        #     d : full sentence embedding dim
        #     D : number of PQ subspaces
        #     K : number of clusters per subspace (n_clusters)
        #     v : value dimension per subspace (v_dim)

        # Args:
        #     x   : (B, L, d) sentence embeddings
        #     tau : Gumbel-Softmax temperature (lower = harder assignments)

        # Returns a dict with keys:
        #     'ste'   : (B, L, D * v_dim)  STE output used for downstream layers
        #     'soft'  : (B, L, D * v_dim)  soft reconstruction (gradient path)
        #     'hard'  : (B, L, D * v_dim)  hard reconstruction (forward value)
        #     'codes' : (B, L, D)          discrete code indices (long tensor)
        
        B, L, _ = x.shape

        # 1) Rotate from original sentence-embedding space to PQ-aligned space.
        #    Input : x      (B, L, d)
        #    Output: x_rot  (B, L, d)
        x_rot = self.rotation(x)                              # (B, L, d)

        # 2) Split each d-dim vector into D independent sub-vectors.
        #    Input : x_rot  (B, L, d)
        #    Output: x_sub  (B, L, D, sub_dim) where sub_dim = d // D
        x_sub = x_rot.view(B, L, self.D, self.sub_dim)       # (B, L, D, sub_dim)

        # 3) Compute assignment logits to each cluster center per subspace.
        #    x_sub   : (B, L, D, sub_dim)
        #    self.K  : (D, K, sub_dim)
        #    logits  : (B, L, D, K)
        logits = torch.einsum('bldi,dki->bldk', x_sub, self.K)  # (B, L, D, n_clusters)

        # 4) Turn logits into soft assignment probabilities.
        #    - train: add Gumbel noise for differentiable categorical sampling
        #    - eval : deterministic softmax
        #    soft_probs: (B, L, D, K), each last-dim slice sums to 1
        if self.training:
            gumbel = -torch.log(
                -torch.log(torch.rand_like(logits).clamp(min=1e-10)) + 1e-10
            )
            soft_probs = F.softmax((logits + gumbel) / tau, dim=-1)
        else:
            soft_probs = F.softmax(logits / tau, dim=-1)      # (B, L, D, n_clusters)

        # 5) Hard assignment (argmax over K clusters) for each (B, L, D) position.
        #    codes: (B, L, D), dtype long, no gradient through argmax.
        codes = logits.argmax(dim=-1)                         # (B, L, D)

        # 6) Hard reconstruction by gathering the selected row from value codebook V.
        #    self.V : (D, K, v)
        #    V_exp  : (B, L, D, K, v)  (broadcast view for gather)
        #    idx    : (B, L, D, 1, v)  (indices expanded along value dim)
        #    hard   : (B, L, D, v)
        V_exp = self.V.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1, -1)
        idx = codes.unsqueeze(-1).unsqueeze(-1).expand(B, L, self.D, 1, self.v_dim)
        hard = V_exp.gather(dim=3, index=idx).squeeze(3)     # (B, L, D, v_dim)

        # 7) Soft reconstruction using weighted average of all K codewords.
        #    soft_probs: (B, L, D, K), self.V: (D, K, v) -> soft: (B, L, D, v)
        soft = torch.einsum('bldk,dkv->bldv', soft_probs, self.V)  # (B, L, D, v_dim)

        # 8) Straight-Through Estimator:
        #    forward value equals hard, backward gradient flows through soft.
        #    ste: (B, L, D, v)
        ste = hard + soft - soft.detach()                     # (B, L, D, v_dim)

        # 9) Flatten (D, v) -> (D * v) to produce GPT-facing token embeddings.
        #    Each returned tensor has shape (B, L, D * v).
        return {
            'ste':   ste.reshape(B, L, self.D * self.v_dim),
            'soft':  soft.reshape(B, L, self.D * self.v_dim),
            'hard':  hard.reshape(B, L, self.D * self.v_dim),
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

    # Config keys specific to RPGUpgrade:
    #     dpq_v_dim (int, optional): Value vector dimension per subspace.
    #         Defaults to n_embd // n_codebook, so DPQ output = n_embd directly
    #         and no projection layer is needed.
    #     quantizer_temperature (float): Initial Gumbel-Softmax temperature τ.
    

    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer,
    ):
        super().__init__(config, dataset, tokenizer)
        print(f'[MODEL] Initializing RPGUpgrade model...')
        # ------------------------------------------------------------------
        # Sentence embedding table  (item_id → sent_emb)
        # TOGGLE: set freeze=True to lock embeddings (original behaviour),
        #         set freeze=False to allow fine-tuning during training.
        # ------------------------------------------------------------------
        FREEZE_SENT_EMB = True   # ← change this line to switch behaviour
        sent_embs_tensor = torch.from_numpy(tokenizer.sent_embs)   # (n_items, d)
        self.sent_emb_table = nn.Embedding.from_pretrained(
            sent_embs_tensor, freeze=FREEZE_SENT_EMB, padding_idx=0
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

        # Gumbel temperature – annealed each epoch via anneal_tau()
        self.gumbel_tau: float = config.get('quantizer_temperature', 1.0)
        self.gumbel_tau_min: float = config.get('min_quantizer_temperature', 0.1)
        self.gumbel_tau_decay: float = config.get('quantizer_temperature_decay', 0.9)

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

    def graph_propagation(self, item_logits: torch.Tensor, n_return_sequences: int):
        
        # Graph-based decoding in item space.

        # Args:
        #     item_logits: (B, n_items-1), scores for item ids 1..n_items-1
        #     n_return_sequences: number of final candidates to return
        
        batch_size = item_logits.shape[0]
        visited_nodes = {batch_id: set() for batch_id in range(batch_size)}

        # Random beam initialization in valid item-id range [1, n_items-1].
        topk_nodes_sorted = torch.randint(
            1, self.dataset.n_items,
            (batch_size, self.num_beams),
            dtype=torch.long,
            device=item_logits.device,
        )

        for batch_id in range(batch_size):
            for node in topk_nodes_sorted[batch_id].detach().cpu().tolist():
                visited_nodes[batch_id].add(node)

        # Iterative expansion + local reranking with DPQ item logits.
        for _ in range(self.propagation_steps):
            all_neighbors = self.adjacency[topk_nodes_sorted].view(batch_size, -1)
            next_nodes = []
            for batch_id in range(batch_size):
                neighbors_in_batch = torch.unique(all_neighbors[batch_id])
                for node in neighbors_in_batch.detach().cpu().tolist():
                    visited_nodes[batch_id].add(node)

                # item_logits columns are 0-based for item ids 1..n_items-1.
                scores = item_logits[batch_id].index_select(0, neighbors_in_batch - 1)
                idxs = torch.topk(scores, self.num_beams).indices
                next_nodes.append(neighbors_in_batch[idxs])

            topk_nodes_sorted = torch.stack(next_nodes, dim=0)

        visited_counts = torch.FloatTensor(
            [[len(visited_nodes[batch_id])] for batch_id in range(batch_size)]
        )
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
        dpq_out = self.dpq(sent_embs, tau=self.gumbel_tau)     
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
        )                                                       # (B, L, D, n_embd)
        outputs.final_states = final_states

        # 5. Loss
        if return_loss:
            assert 'labels' in batch, 'Batch must contain labels.'

            # labels: (B, L) with -100 at ignore positions (padding / no supervision)
            # label_mask (flattened): (B*L,) bool
            label_mask = batch['labels'].view(-1) != -100

            # Flatten token axis and keep only supervised positions:
            # final_states reshape : (B*L, H, E)
            # after mask selection : (M, H, E)
            # average over H heads : (M, E)
            selected = final_states.view(
                -1, self.n_pred_head, self.config['n_embd']
            )[label_mask].mean(dim=1)

            # Project model states into DPQ retrieval space and L2-normalize.
            # query: (M, dpq_out_dim)
            query = F.normalize(self.output_proj(selected), dim=-1)  # (N_valid, dpq_out_dim)

            # Build candidate matrix from ALL item sentence embeddings:
            # sent_emb_table.weight[1:] removes padding id 0.
            # all_sent : (1, I, d) because DPQ expects shape (B, L, d).
            all_sent = self.sent_emb_table.weight[1:].unsqueeze(0)           # (1, n_items-1, d)
            # item_embs after DPQ hard reconstruction: (1, I, dpq_out_dim)
            # squeeze batch axis -> (I, dpq_out_dim), then normalize for cosine scoring
            item_embs = self.dpq(all_sent, tau=self.gumbel_tau)['hard']      # (1, n_items-1, dpq_out_dim)
            item_embs = F.normalize(item_embs.squeeze(0), dim=-1)            # (n_items-1, dpq_out_dim)
            # Dense retrieval scores: (M, dpq_out_dim) @ (dpq_out_dim, I) -> (M, I)
            item_logits = query @ item_embs.T / self.temperature             # (N_valid, n_items-1)

            # CrossEntropy expects class ids in [0, I-1]; subtract 1 to remove padding offset.
            # gt_ids: (M,)
            gt_ids = batch['labels'].view(-1)[label_mask] - 1               # (N_valid,)
            outputs.loss = nn.CrossEntropyLoss()(item_logits, gt_ids)

        return outputs

    def generate(self, batch: dict, n_return_sequences: int = 1, return_loss: bool = False):
        
        # Predict next items.

        # Args:
        #     batch               : Input batch dict.
        #     n_return_sequences  : How many top items to return.
        #     return_loss         : If True, also return the validation loss.

        # Returns:
        #     preds               : (B, n_return_sequences, 1) item IDs
        #     loss (optional)     : scalar tensor, only when return_loss=True
        
        outputs = self.forward(batch, return_loss=return_loss)

        # Gather final hidden state at each sample's last valid timestep:
        # outputs.final_states        : (B, L, H, E)
        # (batch['seq_lens'] - 1)     : (B,) index of each sequence end
        # gathered states             : (B, 1, H, E)
        states = outputs.final_states.gather(
            dim=1,
            index=(batch['seq_lens'] - 1).view(-1, 1, 1, 1).expand(
                -1, 1, self.n_pred_head, self.config['n_embd']
            ),
        )                                                       # (B, 1, D, n_embd)

        # Mean over H heads, project to retrieval space, and normalize:
        # states[:, 0]   : (B, H, E)
        # mean(dim=1)    : (B, E)
        # query          : (B, dpq_out_dim)
        query = self.output_proj(states[:, 0].mean(dim=1))     # (B, dpq_out_dim)
        query = F.normalize(query, dim=-1)

        # Build and normalize candidate item embeddings in the same DPQ space.
        # item_embs      : (I, dpq_out_dim)
        item_embs = self._get_all_item_embs()
        # Similarity scores over all items: (B, dpq_out_dim) @ (dpq_out_dim, I) -> (B, I)
        item_logits = query @ item_embs.T / self.temperature             # (B, n_items-1)

        # Decode with optional graph constraint (same switch semantics as RPG).
        if self.generate_w_decoding_graph:
            if not self.init_flag:
                self.init_graph()
                self.init_flag = True
            preds = self.graph_propagation(
                item_logits=item_logits,
                n_return_sequences=n_return_sequences,
            )
        else:
            # topk indices are 0-based over [0, I-1], add +1 to map back to item ids.
            preds = item_logits.topk(n_return_sequences, dim=-1).indices + 1  # 1-based item IDs
            preds = preds.unsqueeze(-1)                             # (B, n_return_sequences, 1)

        if return_loss:
            return preds, outputs.loss
        return preds
