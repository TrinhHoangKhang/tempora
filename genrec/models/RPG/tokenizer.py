# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

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

    @property
    def n_digit(self):
        """
        Returns the number of digits for the tokenizer.

        The number of digits is determined by the value of `rq_n_codebooks` in the configuration.
        """
        return self.config['n_codebook']

    @property
    def codebook_size(self):
        """
        Returns an integer representing the number of codebooks for the tokenizer.
        """
        return self.config['codebook_size']

    @property
    def max_token_seq_len(self) -> int:
        """
        Returns the maximum token sequence length, including the EOS token.

        Returns:
            int: The maximum token sequence length.
        """
        return self.config['max_item_seq_len']

    @property
    def vocab_size(self) -> int:
        """
        Returns the vocabulary size for the TIGER tokenizer.
        """
        return self.eos_token + 1

    def _get_codebook_bits(self, n_codebook):
        x = math.log2(n_codebook)
        assert x.is_integer() and x >= 0, "Invalid value for n_codebook"
        return int(x)

    def _encode_sent_emb(self, dataset: AbstractDataset, output_path: str):
        """
        Encodes the sentence embeddings for the given dataset and saves them to the specified output path.

        Args:
            dataset (AbstractDataset): The dataset containing the sentences to encode.
            output_path (str): The path to save the encoded sentence embeddings.

        Returns:
            numpy.ndarray: The encoded sentence embeddings.
        """
        assert self.config['metadata'] == 'sentence', \
            'TIGERTokenizer only supports sentence metadata.'

        meta_sentences = [] # 1-base, meta_sentences[0] -> item_id = 1
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
        items_for_training = set()
        for item_seq in dataset.split_data['train']['item_seq']:
            for item in item_seq:
                items_for_training.add(item)
        self.log(f'[TOKENIZER] Items for training: {len(items_for_training)} of {dataset.n_items - 1}')
        mask = np.zeros(dataset.n_items - 1, dtype=bool)
        for item in items_for_training:
            mask[dataset.item2id[item] - 1] = True
        return mask

    def _generate_semantic_id_opq(self, sent_embs, sem_ids_path, train_mask):
        """
        Generates semantic IDs using the OPQ (Optimized Product Quantization) algorithm.
        
        PURPOSE:
        --------
        Convert high-dimensional sentence embeddings (e.g., 384-dim) into compact 32-digit 
        semantic codes (each digit 0-255), where each item is represented as a product of 
        quantizers. This dramatically reduces memory while preserving semantic similarity.
        
        EXAMPLE OUTPUT:
        ---------------
        Input:  "iPhone 13" → sentence embedding (384 floats)
        Output: "iPhone 13" → semantic ID = (45, 123, 67, 200, ..., 189) [32 values, each 0-255]
        
        ALGORITHM OVERVIEW - OPQ (Optimized Product Quantization):
        -----------------------------------------------------------
        1. TRAINING: Learn 32 separate codebooks, each with 256 codewords
           - Uses only training items (train_mask) to learn codebooks
           - Each codebook captures one "view" or "aspect" of the embedding space
        
        2. ENCODING: Quantize ALL items using learned codebooks
           - For each item: find nearest codeword in each of 32 codebooks
           - Result: 32 digits (indices into 32 codebooks)
        
        3. BENEFITS:
           - Embedding (384D) → Semantic ID (32×8bits=256bits) = 97% compression
           - Similarity is preserved: similar embeddings → similar semantic IDs
           - Each digit can independently predict one aspect of item quality

        Args:
            sent_embs (numpy.ndarray): Array of sentence embeddings, shape (n_items, embedding_dim)
                Example: (100000, 384) for 100k items with 384-dim embeddings
            sem_ids_path (str): Path to save the generated semantic IDs as JSON
            train_mask (numpy.ndarray): Boolean mask of shape (n_items-1,), True for training items
                Used to fit codebooks only on training data (prevents data leakage)
        """
        import faiss
        
        # ============ STEP 1: Setup GPU/CPU resources ============
        # faiss is a library for similarity search and vector quantization
        # Decide whether to use GPU acceleration based on config
        if self.config['opq_use_gpu']:
            # GPU setup: allocate resources and enable FP16 for memory efficiency if needed
            res = faiss.StandardGpuResources()
            # Allocate 512 MB temporary GPU memory for internal operations
            res.setTempMemory(1024 * 1024 * 512)
            # Option for GPU<->CPU cloning (copying indices between devices)
            co = faiss.GpuClonerOptions()
            # Use FP16 (half precision) if using many codebooks (n_digit >= 56) to save memory
            # For n_digit=32, this stays as FP32 (full precision)
            co.useFloat16 = self.n_digit >= 56
        
        # CPU threading: use OpenMP for multi-threaded operations
        # From config: typically set to number of CPU cores
        faiss.omp_set_num_threads(self.config['faiss_omp_num_threads'])
        
        # ============ STEP 2: Create OPQ index ============
        # Build the index using the factory string self.index_factory
        # Example factory string: "OPQ32,IVF1,PQ32x8"
        #   - OPQ32: Optimized PQ with 32 codebooks
        #   - IVF1: Inverted File with 1 cluster (no clustering for efficiency)
        #   - PQ32x8: Product Quantization with 32 sub-vectors, 8 bits each (256 codewords per codebook)
        # sent_embs.shape[1] = embedding dimension (e.g., 384)
        # faiss.METRIC_INNER_PRODUCT: similarity metric = dot product (vs L2 distance)
        index = faiss.index_factory(
            sent_embs.shape[1],
            self.index_factory,
            faiss.METRIC_INNER_PRODUCT
        )
        
        self.log(f'[TOKENIZER] Training index...')
        
        # Transfer index to GPU if enabled
        if self.config['opq_use_gpu']:
            index = faiss.index_cpu_to_gpu(res, self.config['opq_gpu_id'], index, co)
        
        # ============ STEP 3: Train codebooks on training data ============
        # This learns the 32 codebooks that will quantize embeddings
        # Only use embeddings where train_mask[i] == True (training items only)
        # sent_embs[train_mask] shape: (num_train_items, embedding_dim)
        # Example: if num_train_items=80000 out of 100000 total
        index.train(sent_embs[train_mask])
        
        # ============ STEP 4: Encode ALL items (train + val + test) ============
        # Now that codebooks are trained, quantize every item's embedding
        # index.add() compresses embeddings into 32-digit codes
        # Input:  sent_embs shape (n_items, embedding_dim) e.g., (100000, 384)
        # Output: stored internally as PQ codes (32 digits per item)
        index.add(sent_embs)
        
        # Transfer index back to CPU for code extraction
        if self.config['opq_use_gpu']:
            index = faiss.index_gpu_to_cpu(index)
        
        # ============ STEP 5: Extract PQ codes from index ============
        # The index now contains quantization codes in binary format
        # We need to extract and decode them
        
        # ivf_index: the underlying inverted file structure
        # IVF organizes codes by cluster (we have 1 cluster, so all items in same partition)
        ivf_index = faiss.downcast_index(index.index)
        
        # invlists: stores actual PQ codes for all items
        # Structure: partitions of items, each partition has binary codes
        invlists = faiss.extract_index_ivf(ivf_index).invlists
        
        # ls = list_size(0): number of items in cluster 0 (should be n_items)
        # Example: ls = 100000 (all items in single cluster since IVF1)
        ls = invlists.list_size(0)
        
        # Extract raw PQ codes from the inverted list
        # get_codes(0): get binary code array for cluster 0
        # invlists.code_size: number of bytes per code (typically 4 bytes = 32 bits for 32 digits × 1 byte each)
        # Example: code_size = 32 bytes (one byte per digit, 256 possible values)
        # pq_codes shape: (ls, code_size) = (100000, 32)
        pq_codes = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
        pq_codes = pq_codes.reshape(-1, invlists.code_size)
        
        # ============ STEP 6: Decode binary PQ codes to digit sequences ============
        # PQ codes are stored as bit-packed bytes; we need to extract individual digits
        # Each digit is n_codebook_bits bits (for 256 values: 8 bits, n_codebook_bits=8)
        
        faiss_sem_ids = []
        n_bytes = pq_codes.shape[1]  # e.g., 32 (one byte per digit)
        
        for u8code in pq_codes:
            # For each item's PQ code (32 bytes of binary data)
            # Create a BitstringReader to extract bits sequentially
            bs = faiss.BitstringReader(faiss.swig_ptr(u8code), n_bytes)
            
            code = []
            # Extract n_digit=32 separate digits from the bitstring
            # Each digit is read as n_codebook_bits bits (e.g., 8 bits → value 0-255)
            for i in range(self.n_digit):
                # read(n_codebook_bits) extracts next 8 bits as an integer
                # Result: integer 0-255 (index into codebook i)
                digit = bs.read(self.n_codebook_bits)
                code.append(digit)
            
            faiss_sem_ids.append(code)
        
        # Convert list of codes to numpy array for easier indexing
        # Shape: (n_items, 32) where each entry is 0-255
        # pq_codes[i][j] = j-th digit code for i-th item
        pq_codes = np.array(faiss_sem_ids)
        
        # ============ STEP 7: Map semantic IDs to item names and save ============
        # Create a dictionary: item_name → 32-digit semantic ID
        # This maps human-readable item names to their quantized codes
        
        item2sem_ids = {}
        for i in range(pq_codes.shape[0]):
            # id2item[i+1]: convert index to item name (offset by 1: id 1 = index 0)
            # pq_codes[i]: the 32-digit code for this item
            item = self.id2item[i + 1]
            # Store as tuple of integers for hashability and JSON serialization
            # Example: "iPhone_13" → (45, 123, 67, ..., 189)
            item2sem_ids[item] = tuple(pq_codes[i].tolist())
        
        self.log(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')
        
        # Save to JSON file for later loading
        # Format: {"item_name": [digit_0, digit_1, ..., digit_31], ...}
        # This file is cached and reused, avoiding re-computation
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
        for item in item2sem_ids:
            tokens = list(item2sem_ids[item])
            for digit in range(self.n_digit):
                # "+ 1" as 0 is reserved for padding
                tokens[digit] += self.codebook_size * digit + 1
            item2sem_ids[item] = tuple(tokens)
        return item2sem_ids

    def _init_tokenizer(self, dataset: AbstractDataset):
        """
        Initialize the tokenizer.

        Args:
            dataset (AbstractDataset): The dataset object.

        Returns:
            dict: A dictionary mapping items to semantic IDs.
        """
        # Load semantic IDs
        sem_ids_path = os.path.join(
            dataset.cache_dir, 'processed',
            f'{os.path.basename(self.config["sent_emb_model"])}_{self.index_factory}.sem_ids'
        )

        if not os.path.exists(sem_ids_path):
            # Load or encode sentence embeddings
            sent_emb_path = os.path.join(
                dataset.cache_dir, 'processed',
                f'{os.path.basename(self.config["sent_emb_model"])}.sent_emb'
            )
            if os.path.exists(sent_emb_path):
                self.log(f'[TOKENIZER] Loading sentence embeddings from {sent_emb_path}...')
                sent_embs = np.fromfile(sent_emb_path, dtype=np.float32).reshape(-1, self.config['sent_emb_dim'])
            else:
                self.log(f'[TOKENIZER] Encoding sentence embeddings...')
                sent_embs = self._encode_sent_emb(dataset, sent_emb_path)
            # PCA
            if self.config['sent_emb_pca'] > 0:
                self.log(f'[TOKENIZER] Applying PCA to sentence embeddings...')
                from sklearn.decomposition import PCA
                pca = PCA(n_components=self.config['sent_emb_pca'], whiten=True)
                sent_embs = pca.fit_transform(sent_embs)
            self.log(f'[TOKENIZER] Sentence embeddings shape: {sent_embs.shape}')

            # Generate semantic IDs
            training_item_mask = self._get_items_for_training(dataset)
            self._generate_semantic_id_opq(sent_embs, sem_ids_path, training_item_mask)

        self.log(f'[TOKENIZER] Loading semantic IDs from {sem_ids_path}...')
        item2sem_ids = json.load(open(sem_ids_path, 'r'))
        item2tokens = self._sem_ids_to_tokens(item2sem_ids)

        return item2tokens

    def _tokenize_first_n_items(self, item_seq: list) -> tuple:
        """
        Tokenizes the first n items in the given item_seq.
        The losses for the first n items can be computed by only forwarding once.
        
        ============ PURPOSE & CONTEXT ============
        This is used in TRAINING mode to create multiple training examples from ONE long item sequence.
        
        Key insight: In a long user history, we want to create multiple overlapping prediction tasks:
          - At position 1: predict item[1] from item[0]
          - At position 2: predict item[2] from item[0:2]
          - At position 3: predict item[3] from item[0:3]
          - ...
          - At position min(len, max_len): predict item[min(len,max_len)] from item[0:min(len,max_len)]
        
        This function creates ONE such training example (the first one).
        
        ============ EXAMPLE WALKTHROUGH ============
        Assume:
          - item_seq = [apple, banana, cherry, date, elderberry]  (5 items)
          - max_token_seq_len = 50
        
        What this function does:
          1. input_ids  = [item_id(apple), item_id(banana), item_id(cherry), item_id(date)]
                        = [1, 2, 3, 4]  (all items EXCEPT last one)
          2. labels     = [item_id(banana), item_id(cherry), item_id(date), item_id(elderberry)]
                        = [2, 3, 4, 5]  (all items EXCEPT first one = shift by 1)
          3. attention_mask = [1, 1, 1, 1]  (mark all as valid, not padding)
          4. seq_lens = 4
        
        Then pad to max_token_seq_len=50:
          5. input_ids  = [1, 2, 3, 4, 0, 0, ..., 0]  (46 zeros appended, total 50)
          6. attention_mask = [1, 1, 1, 1, 0, 0, ..., 0]  (mark padding as ignored)
          7. labels     = [2, 3, 4, 5, -100, -100, ..., -100]  (mark padding with ignored_label=-100)
        
        ============ TRAINING LOSS COMPUTATION ============
        During model training with this batch:
          - Model sees:     input_ids = [1, 2, 3, 4, 0, ..., 0]
          - At position 0:  predict item[1]=2, loss computed ✓
          - At position 1:  predict item[2]=3, loss computed ✓
          - At position 2:  predict item[3]=4, loss computed ✓
          - At position 3:  predict item[4]=5, loss computed ✓
          - At positions 4+: target is -100 (padding), loss is IGNORED (ignore_index=-100)
        
        So from ONE call to this function, we get 4 prediction targets (one per input position).
        This is EFFICIENT: all 4 predictions computed in ONE forward pass!

        Args:
            item_seq (list): The item sequence that contains the first n items.
                Example: [item_0, item_1, item_2, ..., item_k]

        Returns:
            tuple: (input_ids, attention_mask, labels, seq_lens)
                - input_ids: List of item indices [id_0, id_1, ..., id_(k-1), 0, 0, ...]
                - attention_mask: List [1, 1, ..., 1, 0, 0, ...] (1=valid, 0=padding)
                - labels: List [id_1, id_2, ..., id_k, -100, -100, ...]
                - seq_lens: int, actual sequence length (k) before padding
        """
        # ============ STEP 1: Extract input sequence (all items EXCEPT last) ============
        # item_seq[:-1] means "all items except the last one"
        # Example: [apple, banana, cherry, date, elderberry][:-1]
        #        = [apple, banana, cherry, date]
        # For each item, convert its name to its integer ID (1-indexed: item_id >= 1)
        input_ids = [self.item2id[item] for item in item_seq[:-1]]
        
        # ============ STEP 2: Record actual sequence length (before padding) ============
        # This is important: we need to know how long the sequence ACTUALLY was
        # so we can compute losses only for non-padding positions
        # Example: seq_lens = 4 (we have 4 real items before padding)
        seq_lens = len(input_ids)
        
        # ============ STEP 3: Create attention mask for non-padded positions ============
        # attention_mask[i] = 1 if position i is a real token, 0 if it's padding
        # Currently all positions are real, so all 1s
        # Example: attention_mask = [1, 1, 1, 1]
        attention_mask = [1] * seq_lens

        # ============ STEP 4: Pad input_ids and attention_mask to max length ============
        # The transformer expects fixed-size inputs (max_token_seq_len from config: typically 50)
        # Padding token ID is 0 (reserved for padding in vocabulary)
        # Calculate how many padding tokens we need
        pad_lens = self.max_token_seq_len - seq_lens
        # Example: pad_lens = 50 - 4 = 46
        
        # Append padding (zeros) to input_ids
        # Example: input_ids = [1, 2, 3, 4, 0, 0, ..., 0]  (length 50 total)
        input_ids.extend([0] * pad_lens)
        
        # Append padding (zeros) to attention_mask
        # 0 means "ignore this position" (it's padding, not a real token)
        # Example: attention_mask = [1, 1, 1, 1, 0, 0, ..., 0]  (length 50 total)
        attention_mask.extend([0] * pad_lens)

        # ============ STEP 5: Create labels (target items to predict) ============
        # labels[i] = item to predict at position i
        # KEY INSIGHT: labels are SHIFTED by 1 position
        #   - input position 0 → predict item_seq[1]
        #   - input position 1 → predict item_seq[2]
        #   - input position 2 → predict item_seq[3]
        #   - ...
        # This is called "causal" or "autoregressive" prediction: predict NEXT item from current
        # item_seq[1:] means "all items EXCEPT the first one"
        # Example: [apple, banana, cherry, date, elderberry][1:]
        #        = [banana, cherry, date, elderberry]
        # Convert each to item ID: [id(banana), id(cherry), id(date), id(elderberry)] = [2, 3, 4, 5]
        labels = [self.item2id[item] for item in item_seq[1:]]
        
        # ============ STEP 6: Pad labels with ignore_index ============
        # For padding positions, we don't want the loss to be computed (ignore_index=-100)
        # This tells the CrossEntropyLoss function: "don't compute loss here"
        # Example: labels = [2, 3, 4, 5, -100, -100, ..., -100]  (length 50 total)
        labels.extend([self.ignored_label] * pad_lens)

        return input_ids, attention_mask, labels, seq_lens

    def _tokenize_later_items(self, item_seq: list, pad_labels: bool = True) -> tuple:
        """
        Tokenizes the later items in the item sequence.
        Only the last one items are used as the target item.
        
        ============ PURPOSE & CONTEXT ============
        This is used in TRAINING mode (after the first example) to create subsequent training examples.
        
        Key difference from _tokenize_first_n_items():
          - _tokenize_first_n_items: computes loss at ALL non-padding positions (multipl targets)
          - _tokenize_later_items: computes loss ONLY at the LAST position (single target)
        
        Why? To avoid redundant training data:
          - If we already trained on "predict item[1] from item[0]"
          - We don't need to train again on "predict item[1] from item[0]"
          - Instead, use the same input sequence but predict the NEXT item: "predict item[2] from item[0:2]"
        
        ============ SLIDING WINDOW EXAMPLE ============
        Long item sequence: [A, B, C, D, E, F, G]  (length 7)
        max_item_seq_len = 3
        
        _tokenize_first_n_items() call:
          - input_ids = [A, B, C]  (first 3 items)
          - labels = [B, C, D]  (next items)  ← ALL positions produce a loss target
          - Returns 3 training examples in one forward pass
        
        _tokenize_later_items() call 1 (i=1):  item_seq[1:5] = [B, C, D, E]
          - input_ids = [B, C, D]
          - labels = [IGNORE, IGNORE, E]  ← ONLY last position produces a loss target
          - Returns 1 training example
        
        _tokenize_later_items() call 2 (i=2):  item_seq[2:6] = [C, D, E, F]
          - input_ids = [C, D, E]
          - labels = [IGNORE, IGNORE, F]  ← ONLY last position produces a loss target
          - Returns 1 training example
        
        WHY this design?
        ├─ _tokenize_first_n_items: Maximize efficiency for short sequences
        │   └─ If len(item_seq) <= max_len, we only call this once
        │   └─ It produces ALL possible prediction targets in one forward pass
        │
        └─ _tokenize_later_items: Efficiently slide window for long sequences
            └─ If len(item_seq) > max_len, we call this multiple times
            └─ Each call shifts window by 1 position, adds 1 new prediction task
            └─ Only the new (rightmost) position is a new training target
        
        ============ INFERENCE VS TRAINING ============
        In TRAINING mode (pad_labels=True):
          - labels for padding: -100 (ignore_index)
          - We want to learn from this data
        
        In VAL/TEST mode (pad_labels=False):
          - labels for padding: NOT padded (omitted)
          - We're not training, just evaluating the last position
          - No need to pad labels in inference

        Args:
            item_seq (list): The item sequence that we'll extract middle/end items from.
                Example: [item_0, item_1, item_2, item_3, item_4]
            pad_labels (bool): Whether to pad labels with ignore_index.
                - True: used in training, pad labels to max_token_seq_len
                - False: used in validation/testing, don't pad labels (only keep last position)

        Returns:
            tuple: (input_ids, attention_mask, labels, seq_lens)
                - input_ids: List of item indices [id_i, id_(i+1), ..., id_j, 0, 0, ...]
                - attention_mask: List [1, 1, ..., 1, 0, 0, ...] (1=valid, 0=padding)
                - labels: List [IGNORE, IGNORE, ..., id_target] or [id_target] (depends on pad_labels)
                - seq_lens: int, actual sequence length before padding
        """
        # ============ STEP 1: Extract input sequence (all items EXCEPT last) ============
        # item_seq[:-1] means "all items except the last one"
        # Example: if item_seq = [B, C, D, E], then input_ids indices are for [B, C, D]
        input_ids = [self.item2id[item] for item in item_seq[:-1]]
        
        # ============ STEP 2: Record actual sequence length (before padding) ============
        # Example: if input_ids has 3 items, seq_lens = 3
        seq_lens = len(input_ids)
        
        # ============ STEP 3: Create attention mask ============
        # Mark all current (non-padding) positions as valid
        # Example: attention_mask = [1, 1, 1]  (if seq_lens = 3)
        attention_mask = [1] * seq_lens
        
        # ============ STEP 4: Initialize labels with ignore values ============
        # KEY DIFFERENCE from _tokenize_first_n_items():
        # Here, we initialize ALL labels to ignored_label (-100)
        # We'll ONLY set the LAST position to the actual target
        # This means: only the last position produces a loss
        # Example: labels = [-100, -100, -100]  (initially all ignored)
        labels = [self.ignored_label] * seq_lens
        
        # ============ STEP 5: Set ONLY the last position as the prediction target ============
        # item_seq[-1] is the last item in the sequence
        # We want to predict this from the previous items
        # Example: if item_seq = [B, C, D, E], then labels[-1] = item_id(E)
        # So labels = [-100, -100, id(E)]  (only last position matters for loss)
        labels[-1] = self.item2id[item_seq[-1]]

        # ============ STEP 6: Pad to max_token_seq_len (only in training) ============
        pad_lens = self.max_token_seq_len - seq_lens
        # Example: pad_lens = 50 - 3 = 47
        
        # Pad input_ids with zeros (padding token ID)
        # Example: input_ids = [id(B), id(C), id(D), 0, 0, ..., 0]  (total length 50)
        input_ids.extend([0] * pad_lens)
        
        # Pad attention_mask with zeros (mark padding positions as ignored)
        # Example: attention_mask = [1, 1, 1, 0, 0, ..., 0]  (total length 50)
        attention_mask.extend([0] * pad_lens)
        
        # ============ STEP 7: Conditionally pad labels ============
        if pad_labels:
            # TRAINING mode: pad labels with ignore_index
            # Example: labels = [-100, -100, id(E), -100, -100, ..., -100]  (total length 50)
            # This tells the loss function: only position 2 contributes to loss
            labels.extend([self.ignored_label] * pad_lens)
        # else:
        #   VAL/TEST mode: don't pad labels
        #   Labels stay as [-100, -100, id(E)]  (length 3)
        #   This is used for efficient inference (no unnecessary padding)

        return input_ids, attention_mask, labels, seq_lens

    def tokenize_function(self, example: dict, split: str) -> dict:
        """
        Tokenizes the input example based on the specified split.
        
        ============ PURPOSE & CONTEXT ============
        This is the MAIN entry point for converting raw user sequences into training batches.
        
        It's called by the HuggingFace .map() function in the tokenize() method below.
        The function MUST handle TWO different modes:
          1. TRAINING mode (split='train'): Create MULTIPLE training examples from ONE long sequence
          2. INFERENCE mode (split='val' or 'test'): Create SINGLE evaluation example
        
        ============ TRAINING MODE (split='train') ============
        GOAL: Given a long user purchase history, create multiple training examples
        
        EXAMPLE:
          Input sequence: [apple, banana, cherry, date, elderberry, fig, grape] (7 items)
          max_item_seq_len = 3
          
          We want to create training examples:
          Example 1: input=[apple, banana, cherry]      → predict=date      [via _tokenize_first_n_items]
          Example 2: input=[banana, cherry, date]       → predict=elderberry [via _tokenize_later_items]
          Example 3: input=[cherry, date, elderberry]   → predict=fig        [via _tokenize_later_items]
          Example 4: input=[date, elderberry, fig]      → predict=grape      [via _tokenize_later_items]
          
          From ONE user sequence (7 items), we get 4 training examples!
        
        HOW IT WORKS:
          - If len(item_seq) <= max_item_seq_len + 1:
            └─ Call _tokenize_first_n_items ONCE
            └─ This produces multiple targets (all positions except first)
            └─ Example: 4 items + max_len=3 → produces 3 targets in one example
          
          - If len(item_seq) > max_item_seq_len + 1:
            ├─ Call _tokenize_first_n_items ONCE (produces first group of targets)
            └─ Call _tokenize_later_items MULTIPLE TIMES (one target per call)
            └─ Example: 7 items + max_len=3
               ├─ Call 1: items[0:4] → produces 3 targets
               ├─ Call 2: items[1:5] → produces 1 target (only last position)
               ├─ Call 3: items[2:6] → produces 1 target
               └─ Call 4: items[3:7] → produces 1 target
               Total: 3+1+1+1 = 6 training targets
        
        ============ INFERENCE MODE (split='val' or 'test') ============
        GOAL: Evaluate the model on user sequences (typically only last item)
        
        EXAMPLE:
          Input sequence: [apple, banana, cherry, date, elderberry, fig, grape] (7 items)
          max_item_seq_len = 3
          
          We only care about:
            input=[date, elderberry, fig]  (last 3 items)
            target=grape  (last item)
          
          This is what the user "really" just purchased, and we want to predict it
          from their recent history.
        
        ONE call to _tokenize_later_items with:
          - item_seq = items[-(max_len+1):] = items[-4:] = [date, elderberry, fig, grape]
          - pad_labels = False  (don't pad labels in inference)
          - Output: input=[date, elderberry, fig], target=[grape]

        Args:
            example (dict): A single example from the dataset with keys:
                - 'item_seq': list of lists, shape (1, variable_length)
                  └─ Always length 1 (batch_size=1) because this processes one example at a time
                  └─ example['item_seq'][0] is the actual item sequence for one user
                  └─ Example: example = {'item_seq': [[apple, banana, cherry, date, elderberry]]}
            
            split (str): Dataset split identifier
                - 'train': Create MULTIPLE training examples from long sequences
                - 'val' or 'test': Create ONE evaluation example from the last portion

        Returns:
            dict: Contains 4 keys (lists of lists, one per training example generated):
                - 'input_ids': List[List[int]] e.g., [[1, 2, 3, 0, ...], [2, 3, 4, 0, ...]]
                - 'attention_mask': List[List[int]] same shape as input_ids
                - 'labels': List[List[int]] same shape as input_ids
                - 'seq_lens': List[int] actual sequence lengths before padding
                
                IMPORTANT: Each inner list corresponds to ONE training example
                If we create 4 examples, each key has 4 sublists
        """
        # ============ STEP 1: Extract configuration ============
        # Get max sequence length from config (typically 50 from default.yaml)
        max_item_seq_len = self.config['max_item_seq_len']
        
        # ============ STEP 2: Extract the item sequence from example ============
        # example['item_seq'] is a list of lists: [[item_0, item_1, item_2, ...]]
        # We extract the first (and only) list
        # item_seq[0] = [item_0, item_1, item_2, ...]
        item_seq = example['item_seq'][0]
        
        # ============ STEP 3: Handle TRAINING mode ============
        if split == 'train':
            """
            Create multiple training examples from one long sequence.
            
            KEY CALCULATION: n_return_examples
            ─────────────────────────────────
            How many separate training examples should we generate?
            
            Formula: n_return_examples = max(len(item_seq) - max_item_seq_len, 1)
            
            REASONING:
            - If len=10, max_len=5: we can slide window starting at positions 0,1,2,3,4,5
              that's 10-5+1=6 possible starting positions
              but we subtract 1 because _tokenize_first_n_items already covers position 0
              so we need 10-5=5 additional calls to _tokenize_later_items
              total: 1 + 5 = 6 examples ✓
            
            - If len=4, max_len=5: sequence is shorter than window
              10-5=negative, so max(., 1) = 1
              we only call _tokenize_first_n_items once ✓
            
            EXAMPLES:
            - item_seq length 10, max_len=5: n_return_examples = max(10-5, 1) = 5
              └─ _tokenize_first_n_items once (positions 0)
              └─ _tokenize_later_items 4 times (positions 1,2,3,4)
              └─ Total: 5 sliding windows
            
            - item_seq length 4, max_len=5: n_return_examples = max(4-5, 1) = 1
              └─ _tokenize_first_n_items once (covers all items)
              └─ _tokenize_later_items 0 times
              └─ Total: 1 example
            """
            n_return_examples = max(len(item_seq) - max_item_seq_len, 1)

            # ────────── FIRST EXAMPLE: Use _tokenize_first_n_items ──────────
            # For short sequences (len <= max_len + 1), this function produces
            # MULTIPLE prediction targets in a single forward pass
            #
            # item_seq[:min(len(item_seq), max_item_seq_len + 1)]
            # └─ If len <= max_len+1: take all items (e.g., [a,b,c,d] with max_len=5)
            # └─ If len > max_len+1: take first max_len+1 items (e.g., [a,b,c,d,e,f] → [a,b,c,d,e,f])
            #
            # WHY max_len+1? Because:
            #   - _tokenize_first_n_items uses items[:-1] as input
            #   - and items[1:] as labels
            #   - So to have max_len items in input, we need max_len+1 total
            input_ids, attention_mask, labels, seq_lens = self._tokenize_first_n_items(
                item_seq=item_seq[:min(len(item_seq), max_item_seq_len + 1)]
            )
            
            # Store results in lists (one example so far)
            all_input_ids, all_attention_mask, all_labels, all_seq_lens = \
                [input_ids], [attention_mask], [labels], [seq_lens]

            # ────────── SUBSEQUENT EXAMPLES: Use _tokenize_later_items ──────────
            # For long sequences, slide a window across the sequence
            # Each iteration: shift window by 1, create 1 new prediction target
            #
            # Loop from i=1 to i=n_return_examples-1
            # Example: if n_return_examples=4, loop i=1,2,3
            #          (we already did i=0 with _tokenize_first_n_items)
            for i in range(1, n_return_examples):
                # Extract current window: items[i : i+max_item_seq_len+1]
                # Example iteration 1: items[1:6] (5 items starting at position 1)
                # Example iteration 2: items[2:7] (5 items starting at position 2)
                cur_item_seq = item_seq[i:i+max_item_seq_len+1]
                
                # Tokenize this window (only last item produces a loss target)
                input_ids, attention_mask, labels, seq_lens = self._tokenize_later_items(cur_item_seq)
                
                # Append to our lists
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_labels.append(labels)
                all_seq_lens.append(seq_lens)

            # ────────── RETURN ALL EXAMPLES ──────────
            return {
                'input_ids': all_input_ids,        # List of lists: [[...], [...], ...]
                'attention_mask': all_attention_mask,
                'labels': all_labels,
                'seq_lens': all_seq_lens,
            }
        
        # ============ STEP 4: Handle INFERENCE mode (val/test) ============
        else:
            """
            Create ONE evaluation example from the LAST portion of the sequence.
            
            We only care about predicting the last item from recent history.
            
            EXAMPLE:
            - item_seq = [a, b, c, d, e, f, g] (7 items)
            - max_item_seq_len = 3
            - We take items[-(3+1):] = items[-4:] = [d, e, f, g]
            - Input: [d, e, f], Target: [g]
            
            This is what the user "really" just purchased (item g),
            predicted from their recent history (d, e, f).
            
            WHY only recent history?
            - In training, we use all history (to learn complex patterns)
            - In inference, computational efficiency: only need last max_len items
            - Also tests generalization: can model predict from limited context?
            """
            # Extract the LAST (max_item_seq_len+1) items from the sequence
            # If sequence is shorter, take all of it
            # item_seq[-(max_item_seq_len+1):] means "last max_len+1 items"
            #
            # Examples:
            # - item_seq length 100, max_len=3: take items[-4:] = last 4 items
            # - item_seq length 2, max_len=3: take items[-4:] = all 2 items (no wraparound)
            input_ids, attention_mask, labels, seq_lens = self._tokenize_later_items(
                item_seq=item_seq[-(max_item_seq_len+1):],
                pad_labels=False  # Don't pad labels (single target, no need to pad)
            )
            
            # ────────── RETURN SINGLE EXAMPLE ──────────
            # Note: Still return as lists (one inner list each),
            # because downstream code expects batches of examples
            return {
                'input_ids': [input_ids],
                'attention_mask': [attention_mask],
                'labels': [labels[-1:]],  # Only keep the last label (the single prediction target)
                'seq_lens': [seq_lens]
            }

    def tokenize(self, datasets: dict) -> dict:
        """
        Tokenizes the given datasets.
        
        ============ PURPOSE & CONTEXT ============
        This is the TOP-LEVEL function that coordinates tokenization for all dataset splits.
        
        It uses HuggingFace's .map() function to apply tokenize_function() to all examples.
        Then converts everything to PyTorch tensors for efficient training.
        
        ============ WORKFLOW ============
        Input:  datasets = {
                  'train': HuggingFace Dataset with columns ['item_seq', ...]
                  'val': HuggingFace Dataset with columns ['item_seq', ...]
                  'test': HuggingFace Dataset with columns ['item_seq', ...]
                }
        
        Process:
        ────────
        1. For EACH split (train, val, test):
           └─ Apply tokenize_function() to every example
           └─ This converts item sequences → input_ids, attention_mask, labels, seq_lens
        
        2. For EACH split:
           └─ Convert lists → PyTorch tensors
           └─ This enables efficient GPU computation
        
        Output: datasets = {
                  'train': HuggingFace Dataset with PyTorch tensors
                  'val': HuggingFace Dataset with PyTorch tensors
                  'test': HuggingFace Dataset with PyTorch tensors
                }
        
        ============ HuggingFace .map() DETAILS ============
        The .map() function is from HuggingFace Datasets library.
        
        It applies a function to every example in the dataset:
        
        dataset.map(
            function,           # Function to apply to each example
            batched=True,       # Process in batches (but we set batch_size=1)
            batch_size=1,       # Process ONE example at a time (why?)
            remove_columns=..., # Drop original columns after processing
            num_proc=...,       # Use multiple processes for parallelization
            desc=...            # Progress bar description
        )
        
        WHY batch_size=1?
        ──────────────────
        Confusing, right? batched=True but batch_size=1?
        
        REASON: The underlying tokenize_function expects a DICT structure:
          - Input: {'item_seq': [[a, b, c]]}  (list of lists, length 1)
          - Output: {'input_ids': [[...]], 'labels': [[...]]}  (also list of lists)
        
        With batch_size=1:
          ├─ Each call processes 1 example
          └─ batched=True means function receives {'item_seq': [[seq]]}
             not just the raw sequence
        
        This is an awkward API, but it allows the function to return
        MULTIPLE training examples from ONE long sequence (important for training!).
        
        If batch_size > 1, the function would receive batches of sequences,
        making it hard to return different numbers of examples per input sequence.

        Args:
            datasets (dict): A dictionary of datasets to tokenize.
                Keys: 'train', 'val', 'test'
                Values: HuggingFace Dataset objects
                Each dataset has columns: ['item_seq', ...]
                  - 'item_seq': List[List[int]] e.g., [[1, 2, 3, 4, 5]]

        Returns:
            dict: A dictionary of tokenized datasets.
                Keys: 'train', 'val', 'test' (same as input)
                Values: HuggingFace Dataset objects with PyTorch tensors
                Each dataset columns: ['input_ids', 'attention_mask', 'labels', 'seq_lens']
                  - Each column is a PyTorch LongTensor for efficient computation
        """
        # ============ STEP 1: Initialize output dictionary ============
        tokenized_datasets = {}
        
        # ============ STEP 2: Tokenize each split separately ============
        for split in datasets:
            """
            Process one split at a time (train, val, test).
            
            Why separate splits?
            - Training split: creates MULTIPLE examples per sequence (data augmentation)
            - Val/test splits: creates ONE example per sequence (use last window only)
            - So tokenize_function needs to know which split it's processing
            """
            
            # ────────── STEP 2.1: Apply tokenize_function to all examples ──────────
            tokenized_datasets[split] = datasets[split].map(
                # Lambda function: pass the split parameter to tokenize_function
                # datasets[split] will iterate over all examples
                # For each example, call: tokenize_function(example, split=split)
                lambda t: self.tokenize_function(t, split),
                
                # Batch configuration
                batched=True,       # Process examples in batches (required by API)
                batch_size=1,       # But only 1 example per batch (awkward!)
                
                # Column management
                remove_columns=datasets[split].column_names,
                # Remove all original columns (we've converted them)
                # e.g., remove 'item_seq', keep only tokenized outputs
                
                # Parallelization
                num_proc=self.config['num_proc'],
                # Use this many CPU processes for parallel tokenization
                # e.g., num_proc=4 means 4 processes tokenizing in parallel
                # MUCH faster for large datasets!
                
                # Progress bar
                desc=f'Tokenizing {split} set: '
                # Shows progress: "Tokenizing train set: 45%"
            )

        # ============ STEP 3: Convert to PyTorch tensors ============
        # After tokenization, data is still in Python lists
        # Now convert to PyTorch tensors for GPU computation
        for split in datasets:
            """
            set_format(type='torch') converts columns to PyTorch tensors.
            
            BEFORE:
              dataset['train'][0] returns:
              {
                'input_ids': [1, 2, 3, 0, ...],          # Python list
                'attention_mask': [1, 1, 1, 0, ...],
                'labels': [2, 3, 4, -100, ...],
                'seq_lens': 3
              }
            
            AFTER:
              dataset['train'][0] returns:
              {
                'input_ids': tensor([1, 2, 3, 0, ...]),   # PyTorch tensor
                'attention_mask': tensor([1, 1, 1, 0, ...]),
                'labels': tensor([2, 3, 4, -100, ...]),
                'seq_lens': tensor(3)
              }
            
            WHY tensors?
            - Tensors are optimized for numerical computation
            - Can be moved to GPU (GPU-accelerated training)
            - PyTorch DataLoader knows how to handle tensors natively
            - Much faster than Python lists for operations
            """
            tokenized_datasets[split].set_format(type='torch')

        return tokenized_datasets
