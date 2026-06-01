import os
import math
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer


class RPGTokenizer(AbstractTokenizer):
    """
    An example when "codebook_size == 256, n_codebooks == 32":
        0: padding
        1-256: digit 1
        257-512: digit 2
        ...
        7937-8192: digit 32
        8193: eos

    Args:
        config (dict): The configuration dictionary.
        dataset (AbstractDataset): The dataset object.

    Attributes:
        n_codebook_bits (int): The number of bits for the codebook.
        index_factory (str): The index factory name for the OPQ algorithm.
        item2tokens (dict): A dictionary mapping items to their semantic IDs.
        base_user_id (int): The base user ID.
        n_user_tokens (int): The number of user tokens.
        eos_token (int): The end-of-sequence token.
    """
    def __init__(self, config: dict, dataset: AbstractDataset):
        self.n_codebook_bits = self._get_codebook_bits(config['codebook_size'])
        self.index_factory = f'OPQ{config["n_codebook"]},IVF1,PQ{config["n_codebook"]}x{self.n_codebook_bits}'

        super(RPGTokenizer, self).__init__(config, dataset)
        self.item2id = dataset.item2id
        self.user2id = dataset.user2id
        self.id2item = dataset.id_mapping['id2item']
        self.item2tokens = self._init_tokenizer(dataset)
        self.eos_token = self.n_digit * self.codebook_size + 1
        self.ignored_label = -100

        # How many examples (per split) to print detailed window-level debug logs for.
        # Set to 0 to disable. NOTE: with num_proc > 1 the counter lives in a subprocess
        # so counts are per-worker and will exceed this value slightly — still useful.
        self._debug_n_examples = config.get('tokenizer_debug_n_examples', 2)
        self._debug_log_count = {'train': 0, 'val': 0, 'test': 0}

    @property
    def n_digit(self):
        """Number of product quantization digits."""
        return self.config['n_codebook']

    @property
    def codebook_size(self):
        """Number of possible values per digit."""
        return self.config['codebook_size']

    @property
    def max_token_seq_len(self) -> int:
        """Maximum sequence length."""
        return self.config['max_item_seq_len']

    @property
    def vocab_size(self) -> int:
        """Vocabulary size (includes special tokens)."""
        return self.eos_token + 1

    def _get_codebook_bits(self, n_codebook):
        x = math.log2(n_codebook)
        assert x.is_integer() and x >= 0, "Invalid value for n_codebook"
        return int(x)

    def _encode_sent_emb(self, dataset: AbstractDataset, output_path: str):
        """Encode sentence embeddings and save to output_path."""
        assert self.config['metadata'] == 'sentence', \
            'Tokenizer only supports sentence metadata.'
        self.log(f'[TOKENIZER] Encoding sentence embeddings with {self.config["sent_emb_model"]}...')
        
        meta_sentences = [] 
        for i in range(1, dataset.n_items):
            meta_sentences.append(dataset.item2meta[dataset.id_mapping['id2item'][i]])

        if 'sentence-transformers' in self.config['sent_emb_model']:
            sent_emb_model = SentenceTransformer(
                self.config['sent_emb_model']
            ).to(self.config['device'])

            sent_embs = sent_emb_model.encode(
                meta_sentences,
                convert_to_numpy=True,
                batch_size=self.config['sent_emb_batch_size'],
                show_progress_bar=True,
                device=self.config['device']
            )
        elif 'text-embedding-3' in self.config['sent_emb_model']:
            from openai import OpenAI
            client = OpenAI(api_key=self.config['openai_api_key'])

            sent_embs = []
            for i in tqdm(range(0, len(meta_sentences), self.config['sent_emb_batch_size']), desc='Encoding'):
                try:
                    responses = client.embeddings.create(
                        input=meta_sentences[i: i + self.config['sent_emb_batch_size']],
                        model=self.config['sent_emb_model']
                    )
                except:
                    self.log(f'[TOKENIZER] Failed to encode sentence embeddings for {i} - {i + self.config["sent_emb_batch_size"]}')
                    batch = meta_sentences[i: i + self.config['sent_emb_batch_size']]

                    from genrec.utils import num_tokens_from_string
                    new_batch = []
                    for sent in batch:
                        n_tokens = num_tokens_from_string(sent, 'cl100k_base')
                        if n_tokens < 8192:
                            new_batch.append(sent)
                        else:
                            n_chars = 8192 / n_tokens * len(sent) - 100
                            new_batch.append(sent[:int(n_chars)])

                    self.log(f'[TOKENIZER] Retrying with {len(new_batch)} sentences')
                    responses = client.embeddings.create(
                        input=new_batch,
                        model=self.config['sent_emb_model']
                    )

                for response in responses.data:
                    sent_embs.append(response.embedding)
            sent_embs = np.array(sent_embs, dtype=np.float32)


        self.log(f'[TOKENIZER] Saving sentence embeddings to {output_path}...')
        sent_embs.tofile(output_path)
        return sent_embs

    def _get_items_for_training(self, dataset: AbstractDataset) -> np.ndarray:
        """
        Get a boolean mask indicating which items are used for training.

        Args:
            dataset (AbstractDataset): The dataset containing the item sequences.

        Returns:
            np.ndarray: A boolean mask indicating which items are used for training.
        """
        self.log(f'[TOKENIZER] Marking all the IDs of all the items in the training set as True...')
        items_for_training = set()
        for item_seq in dataset.split_data['train']['item_seq']:
            for item in item_seq:
                items_for_training.add(item)
        self.log(f'[TOKENIZER] Items for training: {len(items_for_training)} of {dataset.n_items - 1}')
        mask = np.zeros(dataset.n_items - 1, dtype=bool)
        for item in items_for_training:
            mask[dataset.item2id[item] - 1] = True
        self.log(f"[TOKENIZER] MASK DIMENSION: {mask.shape}")
        return mask

    def _generate_semantic_id_opq(self, sent_embs, sem_ids_path, train_mask):
        """Generate 32-digit semantic codes using OPQ algorithm."""
        import faiss
        
        # Setup GPU/CPU resources
        if self.config['opq_use_gpu']:
            res = faiss.StandardGpuResources()
            res.setTempMemory(1024 * 1024 * 512)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = self.n_digit >= 56
        
        faiss.omp_set_num_threads(self.config['faiss_omp_num_threads'])
        
        # Create OPQ index
        self.log(f'[TOKENIZER] Creating OPQ index with {self.index_factory}...')
        index = faiss.index_factory(
            sent_embs.shape[1],
            self.index_factory,
            faiss.METRIC_INNER_PRODUCT
        )
        
        self.log(f'[TOKENIZER] Training index...')
        
        # Transfer to GPU if enabled
        if self.config['opq_use_gpu']:
            index = faiss.index_cpu_to_gpu(res, self.config['opq_gpu_id'], index, co)
        
        # Train codebooks on training data
        index.train(sent_embs[train_mask])
        
        # Encode all items
        index.add(sent_embs)
        
        # Transfer back to CPU
        if self.config['opq_use_gpu']:
            index = faiss.index_gpu_to_cpu(index)
        
        # Extract PQ codes
        ivf_index = faiss.downcast_index(index.index)
        invlists = faiss.extract_index_ivf(ivf_index).invlists
        ls = invlists.list_size(0)
        
        pq_codes = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
        pq_codes = pq_codes.reshape(-1, invlists.code_size)
        
        # Decode binary PQ codes
        faiss_sem_ids = []
        n_bytes = pq_codes.shape[1]
        
        for u8code in pq_codes:
            bs = faiss.BitstringReader(faiss.swig_ptr(u8code), n_bytes)
            code = []
            for i in range(self.n_digit):
                digit = bs.read(self.n_codebook_bits)
                code.append(digit)
            faiss_sem_ids.append(code)
        
        pq_codes = np.array(faiss_sem_ids)
        
        # Map to item names and save
        item2sem_ids = {}
        for i in range(pq_codes.shape[0]):
            item = self.id2item[i + 1]
            item2sem_ids[item] = tuple(pq_codes[i].tolist())
        
        self.log(f'[TOKENIZER] Saving ITEM2SEM_IDS mapping to {sem_ids_path}...')
        with open(sem_ids_path, 'w') as f:
            json.dump(item2sem_ids, f)

    def _sem_ids_to_tokens(self, item2sem_ids: dict) -> dict:
        """
        Converts semantic IDs to tokens.

        Args:
            item2sem_ids (dict): A dictionary mapping items to their corresponding semantic IDs.

        Returns:
            dict: A dictionary mapping items to their corresponding tokens.
        """
        self.log(f'[TOKENIZER] Converting semantic IDs to tokens...')
        for item in item2sem_ids:
            tokens = list(item2sem_ids[item])
            for digit in range(self.n_digit):
                tokens[digit] += self.codebook_size * digit + 1
            item2sem_ids[item] = tuple(tokens)
        return item2sem_ids

    def _init_tokenizer(self, dataset: AbstractDataset):
        """
        Load or generate semantic IDs for items.
        Return a item2tokens dictionary.
        """
        sem_ids_path = os.path.join(
            dataset.cache_dir, 'processed',
            f'{os.path.basename(self.config["sent_emb_model"])}_{self.index_factory}.sem_ids'
        )

        # If SEMANTIC_IDS CACHE does not exist, generate it.
        self.log(f'[TOKENIZER] ======= ATTTEMPTING TO GENERATE SEMANTIC IDS MAPPING... ========')
        if not os.path.exists(sem_ids_path):
            self.log(f'[TOKENIZER] Semantic IDs mapping does not exist...')
            sent_emb_path = os.path.join(
                dataset.cache_dir, 'processed',
                f'{os.path.basename(self.config["sent_emb_model"])}.sent_emb'
            )

            self.log(f'[TOKENIZER] ======= 1. ATTTEMPTING TO GENERATE SENTENCE EMBEDDINGS ========')
            if os.path.exists(sent_emb_path):
                self.log(f'[TOKENIZER] Sentence embeddings already exist in {sent_emb_path}')
                self.log(f'[TOKENIZER] Loading sentence embeddings from {sent_emb_path}...')
                sent_embs = np.fromfile(sent_emb_path, dtype=np.float32).reshape(-1, self.config['sent_emb_dim'])
            else:
                self.log(f'[TOKENIZER] Creating sentence embeddings...')
                sent_embs = self._encode_sent_emb(dataset, sent_emb_path)
            
            # Apply PCA if configured
            if self.config['sent_emb_pca'] > 0:
                self.log(f'[TOKENIZER] Applying PCA to sentence embeddings...')
                self.log(f'[TOKENIZER] Embeddings shape before PCA: {sent_embs.shape}')
                from sklearn.decomposition import PCA
                pca = PCA(n_components=self.config['sent_emb_pca'], whiten=True)
                sent_embs = pca.fit_transform(sent_embs)
            
            self.log(f'[TOKENIZER] Embeddings shape after PCA: {sent_embs.shape}')
            
            self.log(f'[TOKENIZER] ================== 2. GENERATE TRAINING ITEM MASK =============')
            training_item_mask = self._get_items_for_training(dataset)
            
            self.log(f'[TOKENIZER] ================== 3. GENERATE SEMANTIC IDS ===================')
            self._generate_semantic_id_opq(sent_embs, sem_ids_path, training_item_mask)
        else:
            self.log(f'[TOKENIZER] Semantic IDs mapping already exists in {sem_ids_path}')
            
        # LOAD SEMANTIC IDS FROM CACHE.
        self.log(f'[TOKENIZER] =============== CREATING ITEM2TOKENS MAPPING... ==============')
        self.log(f'[TOKENIZER] Loading ITEM2SEM_IDS mapping from {sem_ids_path}...')
        item2sem_ids = json.load(open(sem_ids_path, 'r'))
        # CONVERT SEMANTIC IDS TO TOKENS.
        item2tokens = self._sem_ids_to_tokens(item2sem_ids)
        self.log(f'[TOKENIZER] Successfully created ITEM2TOKENS mapping...')
        return item2tokens

    def _tokenize_first_n_items(self, item_seq: list) -> tuple:
        """Tokenize first n items. Creates training examples with inputs shifted by 1."""
        input_ids = [self.item2id[item] for item in item_seq[:-1]]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens
        
        pad_lens = self.max_token_seq_len - seq_lens
        input_ids.extend([0] * pad_lens)
        attention_mask.extend([0] * pad_lens)

        labels = [self.item2id[item] for item in item_seq[1:]]
        labels.extend([self.ignored_label] * pad_lens)

        return input_ids, attention_mask, labels, seq_lens

    def _tokenize_later_items(self, item_seq: list, pad_labels: bool = True) -> tuple:
        """Tokenize middle/end items. Only last position produces loss."""
        input_ids = [self.item2id[item] for item in item_seq[:-1]]
        seq_lens = len(input_ids)
        attention_mask = [1] * seq_lens
        labels = [self.ignored_label] * seq_lens
        labels[-1] = self.item2id[item_seq[-1]]

        pad_lens = self.max_token_seq_len - seq_lens
        input_ids.extend([0] * pad_lens)
        attention_mask.extend([0] * pad_lens)
        
        if pad_labels:
            labels.extend([self.ignored_label] * pad_lens)

        return input_ids, attention_mask, labels, seq_lens

    def _debug_fmt_ids(self, ids: list, label: str = '', limit: int = 8) -> str:
        """Format a token id list for compact debug printing (truncates long lists)."""
        if len(ids) <= limit:
            preview = str(ids)
        else:
            preview = f'[{", ".join(str(x) for x in ids[:limit])}, ... ({len(ids)} total)]'
        return f'{label}{preview}' if label else preview

    def tokenize_function(self, example: dict, split: str) -> dict:
        """Tokenize example: sliding windows for training, last window for inference."""
        max_item_seq_len = self.config['max_item_seq_len']
        item_seq = example['item_seq'][0]

        should_log = (self._debug_n_examples > 0 and
                      self._debug_log_count[split] < self._debug_n_examples)

        # TRAINING mode: create multiple examples via sliding window
        if split == 'train':

            n_return_examples = max(len(item_seq) - max_item_seq_len, 1)

            if should_log:
                self.log(
                    f'\n[TOKENIZER:DEBUG] ── TRAIN example #{self._debug_log_count[split] + 1} ──────────────────────────\n'
                    f'[TOKENIZER:DEBUG]  item_seq (len={len(item_seq)}): {list(item_seq)}\n'
                    f'[TOKENIZER:DEBUG]  max_item_seq_len = {max_item_seq_len}\n'
                    f'[TOKENIZER:DEBUG]  n_return_examples = max(len({len(item_seq)}) - {max_item_seq_len}, 1) = {n_return_examples}\n'
                    f'[TOKENIZER:DEBUG]  → This one user will produce {n_return_examples} training row(s)'
                )

            # Window 0 — "first n items": ALL positions generate a label (next-item prediction
            # across the whole prefix).  Slice used: item_seq[0 : min(len, max_len+1)]
            first_slice = item_seq[:min(len(item_seq), max_item_seq_len + 1)]
            input_ids, attention_mask, labels, seq_lens = self._tokenize_first_n_items(
                item_seq=first_slice
            )

            if should_log:
                self.log(
                    f'[TOKENIZER:DEBUG]  Window 0 (_tokenize_first_n_items)\n'
                    f'[TOKENIZER:DEBUG]    slice  : item_seq[0:{min(len(item_seq), max_item_seq_len + 1)}] = {list(first_slice)}\n'
                    f'[TOKENIZER:DEBUG]    input  : {self._debug_fmt_ids(input_ids)}  '
                    f'← all items except the last (ids via item2id)\n'
                    f'[TOKENIZER:DEBUG]    labels : {self._debug_fmt_ids(labels)}  '
                    f'← shifted by 1 (every position predicts the next item); '
                    f'{self.ignored_label} = ignored (padding)\n'
                    f'[TOKENIZER:DEBUG]    attn   : {self._debug_fmt_ids(attention_mask)}  '
                    f'← 1=real token, 0=pad\n'
                    f'[TOKENIZER:DEBUG]    seq_len: {seq_lens}  ← number of real (non-pad) tokens'
                )

            all_input_ids, all_attention_mask, all_labels, all_seq_lens = \
                [input_ids], [attention_mask], [labels], [seq_lens]

            # Windows 1..n — "later items": ONLY the last position generates a label.
            # The window slides one step to the right for each i.
            for i in range(1, n_return_examples):
                cur_item_seq = item_seq[i:i + max_item_seq_len + 1]
                input_ids, attention_mask, labels, seq_lens = self._tokenize_later_items(cur_item_seq)
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_labels.append(labels)
                all_seq_lens.append(seq_lens)

                if should_log:
                    self.log(
                        f'[TOKENIZER:DEBUG]  Window {i} (_tokenize_later_items)\n'
                        f'[TOKENIZER:DEBUG]    slice  : item_seq[{i}:{i + max_item_seq_len + 1}] = {list(cur_item_seq)}\n'
                        f'[TOKENIZER:DEBUG]    input  : {self._debug_fmt_ids(input_ids)}\n'
                        f'[TOKENIZER:DEBUG]    labels : {self._debug_fmt_ids(labels)}  '
                        f'← only the LAST position is a real label '
                        f'(avoids re-training on already-seen history)\n'
                        f'[TOKENIZER:DEBUG]    attn   : {self._debug_fmt_ids(attention_mask)}\n'
                        f'[TOKENIZER:DEBUG]    seq_len: {seq_lens}'
                    )

            if should_log:
                self._debug_log_count[split] += 1
                self.log(
                    f'[TOKENIZER:DEBUG]  → Produced {len(all_input_ids)} row(s) for this user\n'
                    f'[TOKENIZER:DEBUG] ────────────────────────────────────────────────────────'
                )

            return {
                'input_ids': all_input_ids,
                'attention_mask': all_attention_mask,
                'labels': all_labels,
                'seq_lens': all_seq_lens,
            }

        # INFERENCE mode (val / test): only the LAST window is used.
        # Target item is always the last element of item_seq (left out during splitting).
        else:
            inference_slice = item_seq[-(max_item_seq_len + 1):]
            input_ids, attention_mask, labels, seq_lens = self._tokenize_later_items(
                item_seq=inference_slice,
                pad_labels=True
            )

            if should_log:
                self.log(
                    f'\n[TOKENIZER:DEBUG] ── {split.upper()} example #{self._debug_log_count[split] + 1} ──────────────────────────\n'
                    f'[TOKENIZER:DEBUG]  item_seq (len={len(item_seq)}): {list(item_seq)}\n'
                    f'[TOKENIZER:DEBUG]  Using LAST window only (inference mode)\n'
                    f'[TOKENIZER:DEBUG]    slice  : item_seq[-{max_item_seq_len + 1}:] = {list(inference_slice)}\n'
                    f'[TOKENIZER:DEBUG]    input  : {self._debug_fmt_ids(input_ids)}\n'
                    f'[TOKENIZER:DEBUG]    labels : {self._debug_fmt_ids(labels)}  '
                    f'← only the very last position carries the ground-truth target\n'
                    f'[TOKENIZER:DEBUG]    attn   : {self._debug_fmt_ids(attention_mask)}\n'
                    f'[TOKENIZER:DEBUG]    seq_len: {seq_lens}\n'
                    f'[TOKENIZER:DEBUG] ────────────────────────────────────────────────────────'
                )
                self._debug_log_count[split] += 1

            return {
                'input_ids': [input_ids],
                'attention_mask': [attention_mask],
                'labels': [labels],
                'seq_lens': [seq_lens]
            }

    def tokenize(self, datasets: dict) -> dict:
        """Apply tokenize_function to all splits and convert to PyTorch tensors."""
        tokenized_datasets = {}

        self.log(
            f'\n[TOKENIZER] ══════════════════════════════════════════════════════\n'
            f'[TOKENIZER]  TOKENIZE() — overview of what is about to happen\n'
            f'[TOKENIZER]  max_item_seq_len = {self.config["max_item_seq_len"]}\n'
            f'[TOKENIZER]  Training strategy  : SLIDING WINDOW\n'
            f'[TOKENIZER]    • Window 0  → _tokenize_first_n_items  (ALL positions labelled)\n'
            f'[TOKENIZER]    • Window 1+ → _tokenize_later_items    (LAST position only)\n'
            f'[TOKENIZER]  Inference strategy : LAST WINDOW ONLY (_tokenize_later_items)\n'
            f'[TOKENIZER]  Detailed window logs enabled for first '
            f'{self._debug_n_examples} example(s) per split '
            f'(set tokenizer_debug_n_examples=0 to silence)\n'
            f'[TOKENIZER] ══════════════════════════════════════════════════════'
        )

        for split in datasets:
            n_users = len(datasets[split])
            self.log(
                f'[TOKENIZER] Tokenizing {split} set '
                f'({n_users} user sequences)...'
            )
            # Reset per-split counter so debug logs fire at the start of each split
            self._debug_log_count[split] = 0

            tokenized_datasets[split] = datasets[split].map(
                lambda t: self.tokenize_function(t, split),
                batched=True,
                batch_size=1,
                remove_columns=datasets[split].column_names,
                num_proc=self.config['num_proc'],
                desc=f'Tokenizing {split} set: '
            )

            n_rows = len(tokenized_datasets[split])
            expansion = n_rows / n_users if n_users > 0 else 0
            self.log(
                f'[TOKENIZER] {split} set tokenized: '
                f'{n_users} users → {n_rows} rows '
                f'(expansion factor ≈ {expansion:.2f}x via sliding window)'
            )

        # Convert to PyTorch tensors for efficient GPU computation
        for split in datasets:
            tokenized_datasets[split].set_format(type='torch')

        # Showing the content of the tokenized datasets.
        self.log("[TOKENIZER] CONTENT OF TOKENIZED DATASETS:")
        for split in tokenized_datasets:
            self.log(f"[TOKENIZER] {split}: {tokenized_datasets[split]}")
        self.log("[TOKENIZER] ONE EXAMPLE: ")
        self.log(f"[TOKENIZER] {tokenized_datasets['train'][99]}")
            
        return tokenized_datasets
