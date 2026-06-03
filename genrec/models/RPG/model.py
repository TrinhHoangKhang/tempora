import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Config, GPT2Model
from genrec.dataset import AbstractDataset
from genrec.model import AbstractModel
from genrec.tokenizer import AbstractTokenizer


class ResBlock(nn.Module):
    """Residual block: x + SiLU(Linear(x)). Initialized as identity."""

    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        torch.nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()

    def forward(self, x):
        """Apply residual connection: x + SiLU(Linear(x))."""
        return x + self.act(self.linear(x))


class RPG(AbstractModel):
    """GPT-2 Sequential Recommendation with 32-digit Product Quantization."""
    
    def __init__(
        self,
        config: dict,
        dataset: AbstractDataset,
        tokenizer: AbstractTokenizer
    ):
        """Initialize RPG model with GPT-2 backbone and 32 prediction heads."""
        super(RPG, self).__init__(config, dataset, tokenizer)

        # Item ID -> 32 semantic digit codes
        self.item_id2tokens = self._map_item_tokens().to(self.config['device'])

        # Initialize GPT-2 encoder
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

        # 32 prediction heads (one per digit)
        self.n_pred_head = self.tokenizer.n_digit
        pred_head_list = []
        for i in range(self.n_pred_head):
            pred_head_list.append(ResBlock(self.config['n_embd']))
        self.pred_heads = nn.Sequential(*pred_head_list)

        # Loss and generation setup
        self.temperature = self.config['temperature']
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.ignored_label)
        self.generate_w_decoding_graph = False
        self.init_flag = False
        self.chunk_size = config['chunk_size']
        self.num_beams = config['num_beams']
        self.n_edges = config['n_edges']
        self.propagation_steps = config['propagation_steps']

    def _map_item_tokens(self) -> torch.Tensor:
        """Create lookup table: item_id -> 32-digit semantic code."""
        item_id2tokens = torch.zeros((self.dataset.n_items, self.tokenizer.n_digit), dtype=torch.long)
        for item in self.tokenizer.item2tokens:
            item_id = self.dataset.item2id[item]
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
        """Predict 32-digit codes for next item in sequence. Compute loss if return_loss=True."""
        
        # Token lookup and embedding
        input_tokens = self.item_id2tokens[batch['input_ids']]
        input_embs = self.gpt2.wte(input_tokens).mean(dim=-2)
        
        # GPT-2 encoding
        outputs = self.gpt2(
            inputs_embeds=input_embs,
            attention_mask=batch['attention_mask']
        )
        
        # Apply 32 prediction heads
        final_states = [self.pred_heads[i](outputs.last_hidden_state).unsqueeze(-2) 
                       for i in range(self.n_pred_head)]
        final_states = torch.cat(final_states, dim=-2)
        outputs.final_states = final_states
        
        # Loss computation
        if return_loss:
            assert 'labels' in batch, 'The batch must contain the labels.'
            
            # Create mask for valid positions (non-padding)
            label_mask = batch['labels'].view(-1) != -100
            
            # Extract valid predictions
            selected_states = final_states.view(-1, self.n_pred_head, self.config['n_embd'])[label_mask]
            
            # Normalize to unit sphere
            selected_states = F.normalize(selected_states, dim=-1)
            
            # Split by digit
            selected_states = torch.chunk(selected_states, self.n_pred_head, dim=1)
            
            # Extract token embeddings and split by digit
            token_emb = self.gpt2.wte.weight[1:-1]
            token_emb = F.normalize(token_emb, dim=-1)
            token_embs = torch.chunk(token_emb, self.n_pred_head, dim=0)
            
            # Compute logits and loss for each digit
            token_logits = [
                torch.matmul(selected_states[i].squeeze(dim=1), token_embs[i].T) / self.temperature
                for i in range(self.n_pred_head)
            ]
            
            # Extract ground-truth digit codes
            token_labels = self.item_id2tokens[batch['labels'].view(-1)[label_mask]]
            
            # Compute loss for each digit
            losses = [
                self.loss_fct(
                    token_logits[i],
                    token_labels[:, i] - i * self.config['codebook_size'] - 1
                )
                for i in range(self.n_pred_head)
            ]
            outputs.loss = torch.mean(torch.stack(losses))
        
        return outputs

    def build_ii_sim_mat(self):
        """Build item-item similarity matrix from semantic token embeddings."""
        n_items = self.dataset.n_items
        n_digit = self.tokenizer.n_digit
        codebook_size = self.tokenizer.codebook_size

        # Extract and reshape token embeddings: (vocab, dim) -> (n_digit, codebook_size, dim)
        token_embs = self.gpt2.wte.weight[1:-1].view(n_digit, codebook_size, -1)
        token_embs = F.normalize(token_embs, dim=-1)
        
        # Compute per-digit similarity matrices: (n_digit, codebook_size, codebook_size)
        token_sims = torch.bmm(token_embs, token_embs.transpose(1, 2))
        token_sims_01 = 0.5 * (token_sims + 1.0)

        # Fill item-item similarity matrix in chunks
        item_item_sim = torch.zeros((n_items, n_items), device=self.gpt2.device, dtype=torch.float32)

        for i_start in range(1, n_items, self.chunk_size):
            i_end = min(i_start + self.chunk_size, n_items)
            tokens_i = self.item_id2tokens[i_start:i_end]

            for j_start in range(1, n_items, self.chunk_size):
                j_end = min(j_start + self.chunk_size, n_items)
                tokens_j = self.item_id2tokens[j_start:j_end]

                block_size_i = i_end - i_start
                block_size_j = j_end - j_start
                sum_block = torch.zeros((block_size_i, block_size_j), device=self.gpt2.device, dtype=torch.float32)

                # Accumulate similarity across all digits
                for k in range(n_digit):
                    row_inds = tokens_i[:, k] - k * codebook_size - 1
                    col_inds = tokens_j[:, k] - k * codebook_size - 1
                    temp = token_sims_01[k].index_select(0, row_inds).index_select(1, col_inds)
                    sum_block += temp

                item_item_sim[i_start:i_end, j_start:j_end] = sum_block / n_digit

        return item_item_sim

    def build_adjacency_list(self, item_item_sim):
        """Find top n_edges nearest neighbors for each item."""
        return torch.topk(item_item_sim, k=self.n_edges, dim=-1).indices

    def init_graph(self):
        """Initialize k-NN graph for graph-constrained decoding."""
        self.tokenizer.log("Building item-item similarity matrix...")
        item_item_sim = self.build_ii_sim_mat()
        self.adjacency = self.build_adjacency_list(item_item_sim)
        self.tokenizer.log("Graph initialized.")

    def graph_propagation(self, token_logits, n_return_sequences):
        """Graph-based search constrained by k-NN graph."""
        batch_size = token_logits.shape[0]
        visited_nodes = {}
        for batch_id in range(batch_size):
            visited_nodes[batch_id] = set()

        # Random initialization
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

        # Iterative graph traversal
        for sid in range(self.propagation_steps):
            all_neighbors = self.adjacency[topk_nodes_sorted].view(batch_size, -1)

            next_nodes = []
            for batch_id in range(batch_size):
                neighbors_in_batch = torch.unique(all_neighbors[batch_id])
                for node in neighbors_in_batch.cpu().numpy().tolist():
                    visited_nodes[batch_id].add(node)

                # Score neighbors
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
        return topk_nodes_sorted[:,:n_return_sequences].unsqueeze(-1), visited_counts

    def generate(self, batch, n_return_sequences=1, return_loss=False):
        """Predict next items. Use graph search if generate_w_decoding_graph=True, else direct top-k.
        
        Args:
            batch: Input batch
            n_return_sequences: Number of top items to return
            return_loss: If True, also return validation loss
        
        Returns:
            preds: Predicted item IDs
            loss (optional): Validation loss if return_loss=True
        """
        # Forward pass
        outputs = self.forward(batch, return_loss=return_loss)
        
        # Extract last state and normalize
        states = outputs.final_states.gather(
            dim=1,
            index=(batch['seq_lens'] - 1).view(-1, 1, 1, 1).expand(-1, 1, self.n_pred_head, self.config['n_embd'])
        )
        states = F.normalize(states, dim=-1)
        
        # Compute token logits
        token_emb = self.gpt2.wte.weight[1:-1]
        token_emb = F.normalize(token_emb, dim=-1)
        token_embs = torch.chunk(token_emb, self.n_pred_head, dim=0)
        
        logits = [torch.matmul(states[:,0,i,:], token_embs[i].T) / self.temperature 
                 for i in range(self.n_pred_head)]
        logits = [F.log_softmax(logit, dim=-1) for logit in logits]
        token_logits = torch.cat(logits, dim=-1)
        
        # Decode items
        if self.generate_w_decoding_graph:
            if not self.init_flag:
                self.init_graph()
                self.init_flag = True
            preds = self.graph_propagation(token_logits=token_logits, n_return_sequences=n_return_sequences)
        else:
            # Direct greedy decoding
            item_logits = torch.gather(
                input=token_logits.unsqueeze(-2).expand(-1, self.dataset.n_items, -1),
                dim=-1,
                index=(self.item_id2tokens[1:,:] - 1).unsqueeze(0).expand(token_logits.shape[0], -1, -1)
            ).mean(dim=-1)
            preds = item_logits.topk(n_return_sequences, dim=-1).indices + 1
            preds = preds.unsqueeze(-1)
        
        if return_loss:
            return preds, outputs.loss
        else:
            return preds