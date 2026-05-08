# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Upgraded RPG Tokenizer with Differentiable OPQ Support

This tokenizer extends the original RPGTokenizer with:
1. OPQ Rotation Matrix Extraction: Extract the learned rotation matrix R from faiss index
2. PQ Codebook Extraction: Extract the learned codebook vectors K from faiss index
3. Differentiable OPQ: Enable fine-tuning of quantization parameters via backprop

Key differences from original:
- Stores the trained faiss index for parameter extraction
- Provides utility functions to convert FAISS OPQ parameters to PyTorch tensors
- Enables gradient-based optimization of the quantization layer
"""

import os
import math
import json
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from genrec.dataset import AbstractDataset
from genrec.tokenizer import AbstractTokenizer


def extract_opq_rotation_matrix(faiss_index) -> torch.Tensor:
    """
    Extract the OPQ rotation matrix R from a trained faiss index.
    
    FAISS OPQ stores a learned rotation matrix that projects embeddings to a space
    where product quantization works better (lower reconstruction error).
    
    Args:
        faiss_index: A trained faiss.index_factory object with OPQ algorithm.
                    Must be of form 'OPQ{D},IVF1,PQ{D}x{b}' where D is n_codebook.
    
    Returns:
        torch.Tensor: Rotation matrix R of shape (embedding_dim, embedding_dim).
                     This is an orthogonal matrix that should be used as the initial
                     weights for the differentiable rotation layer.
    
    Raises:
        RuntimeError: If index does not contain OPQ component or is not trained.
    
    Technical Details:
        - FAISS stores the OPQ transformation as a LinearTransform object
        - We extract this and convert to numpy, then to torch
        - The rotation matrix satisfies: R @ R^T = I (orthogonal constraint)
    """
    try:
        import faiss
        # Get the VectorTransform object (contains the OPQ rotation)
        vt = faiss.downcast_VectorTransform(faiss_index.chain.at(0))
        
        if vt is None:
            raise RuntimeError("No VectorTransform (OPQ) found in index chain")
        
        # Extract rotation matrix as numpy array
        # Shape: (embedding_dim, embedding_dim)
        rotation_matrix_np = faiss.vector_to_array(vt.A).reshape(vt.d_out, vt.d_in)
        
        # Convert to torch tensor
        rotation_matrix = torch.from_numpy(rotation_matrix_np.T.copy()).float()
        
        return rotation_matrix
    except Exception as e:
        raise RuntimeError(f"Failed to extract OPQ rotation matrix: {str(e)}")


def extract_pq_codebooks(faiss_index, embedding_dim: int, n_codebook: int, codebook_size: int) -> torch.Tensor:
    """
    Extract the PQ (Product Quantization) codebooks from a trained faiss index.
    
    PQ codebooks are the learned centroids for each subspace. Each subspace has
    codebook_size centroids of dimension (embedding_dim / n_codebook).
    
    Args:
        faiss_index: A trained faiss.index_factory object with OPQ+PQ components.
        embedding_dim: Original embedding dimension.
        n_codebook: Number of subspaces (typically 32).
        codebook_size: Number of centroids per subspace (typically 256).
    
    Returns:
        torch.Tensor: Codebooks of shape (n_codebook, codebook_size, embedding_dim // n_codebook).
                     codebooks[d] contains all centroids for subspace d.
    
    Raises:
        RuntimeError: If codebooks cannot be extracted from index.
    
    Technical Details:
        - PQ codebooks are stored in the IndexIVF's ProductQuantizer
        - We extract the centroid table and reshape it to (D, K, d/D) format
        - Each centroid is a vector in the subspace: shape (embedding_dim // n_codebook,)
    """
    try:
        import faiss
        # Get the IVF index
        ivf_index = faiss.downcast_index(faiss_index.index)
        
        # Get the ProductQuantizer from the IVF index
        pq = faiss.downcast_index(ivf_index).pq if hasattr(ivf_index, 'pq') else None
        
        if pq is None:
            raise RuntimeError("No ProductQuantizer found in index")
        
        # Extract codebook centroids
        # Shape: (n_codebook * codebook_size, embedding_dim // n_codebook)
        codebooks_np = faiss.vector_to_array(pq.centroids).reshape(
            n_codebook, codebook_size, embedding_dim // n_codebook
        )
        
        # Convert to torch tensor
        codebooks = torch.from_numpy(codebooks_np.copy()).float()
        
        return codebooks
    except Exception as e:
        raise RuntimeError(f"Failed to extract PQ codebooks: {str(e)}")


class RPGTokenizerUpgrade(AbstractTokenizer):
    """
    Upgraded RPG Tokenizer with Differentiable OPQ Support.
    
    Extends the original RPGTokenizer to enable fine-tuning of the OPQ rotation matrix
    and PQ codebooks through gradient-based optimization.
    
    Architecture:
        Input Embedding (sent_embs)
            ↓
        [Optional: PCA] (from config)
            ↓
        OPQ Rotation Matrix R (learned in FAISS, extracted here)
            ↓
        Product Quantization
            ├─ Subspace 1: Project to d/D dims, quantize with K centroids
            ├─ Subspace 2: ...
            ├─ ...
            └─ Subspace D: ...
            ↓
        Semantic ID (D-digit code, each digit ∈ [0, K-1])
    
    Storage:
        - self.opq_rotation: Extracted rotation matrix (for DifferentiableOPQ layer)
        - self.pq_codebooks: Extracted codebook centroids (for DifferentiableOPQ layer)
    
    Training Mode:
        When the model is in training mode, these parameters can be optimized
        through the DifferentiableOPQ layer using Gumbel-Softmax and STE.
    """
    
    def __init__(self, config: dict, dataset: AbstractDataset):
        self.n_codebook_bits = self._get_codebook_bits(config['codebook_size'])
        self.index_factory = f'OPQ{config["n_codebook"]},IVF1,PQ{config["n_codebook"]}x{self.n_codebook_bits}'
        
        super(RPGTokenizerUpgrade, self).__init__(config, dataset)
        self.item2id = dataset.item2id
        self.user2id = dataset.user2id
        self.id2item = dataset.id_mapping['id2item']
        self.item2tokens = self._init_tokenizer(dataset)
        self.eos_token = self.n_digit * self.codebook_size + 1
        self.ignored_label = -100
        
        # Store OPQ parameters for differentiable optimization
        self.opq_rotation = None  # Will be populated when faiss index is available
        self.pq_codebooks = None  # Will be populated when faiss index is available
        self.faiss_index = None   # Keep reference to faiss index for extraction

    @property
    def n_digit(self):
        """Number of codebooks (subspaces)."""
        return self.config['n_codebook']

    @property
    def codebook_size(self):
        """Number of centroids per codebook."""
        return self.config['codebook_size']

    @property
    def max_token_seq_len(self) -> int:
        """Maximum token sequence length."""
        return self.config['max_item_seq_len']

    @property
    def vocab_size(self) -> int:
        """
        Returns the vocabulary size for the TIGER tokenizer.
        
        Vocab layout:
        - Token 0: Padding
        - Tokens 1 to codebook_size: Digit 1 values
        - Tokens (codebook_size+1) to (2*codebook_size): Digit 2 values
        - ...
        - Tokens (n_digit-1)*codebook_size+1 to n_digit*codebook_size: Digit n_digit values
        - Token (n_digit*codebook_size+1): EOS
        """
        return self.eos_token + 1

    def _get_codebook_bits(self, n_codebook):
        """Calculate the number of bits needed to represent n_codebook values."""
        x = math.log2(n_codebook)
        assert x.is_integer() and x >= 0, "Invalid value for n_codebook"
        return int(x)

    def _encode_sent_emb(self, dataset: AbstractDataset, output_path: str):
        """Encodes sentence embeddings using SentenceTransformer or OpenAI API."""
        assert self.config['metadata'] == 'sentence', \
            'RPGTokenizerUpgrade only supports sentence metadata.'

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
                    self.log(f'[TOKENIZER] Failed to encode sentence embeddings for {i}')
                    batch = meta_sentences[i: i + self.config['sent_emb_batch_size']]

                    from genrec.utils import num_tokens_from_string
                    new_batch = []
                    for sent in batch:
                        tokens = num_tokens_from_string(sent, "cl100k_base")
                        if tokens > 8191:
                            sent = sent[:int(len(sent) * (8191 / tokens))]
                        new_batch.append(sent)

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
        """Get items that appear in training split for OPQ training."""
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
        """Generates semantic IDs using FAISS OPQ algorithm and extracts parameters."""
        import faiss
        
        if self.config['opq_use_gpu']:
            res = faiss.StandardGpuResources()
            res.setTempMemory(1024 * 1024 * 512)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = self.n_digit >= 56
        
        faiss.omp_set_num_threads(self.config['faiss_omp_num_threads'])
        index = faiss.index_factory(
            sent_embs.shape[1],
            self.index_factory,
            faiss.METRIC_INNER_PRODUCT
        )
        
        self.log(f'[TOKENIZER] Training OPQ index...')
        if self.config['opq_use_gpu']:
            index = faiss.index_cpu_to_gpu(res, self.config['opq_gpu_id'], index, co)
        
        index.train(sent_embs[train_mask])
        index.add(sent_embs)
        
        if self.config['opq_use_gpu']:
            index = faiss.index_gpu_to_cpu(index)
        
        # Store faiss index for later parameter extraction
        self.faiss_index = index
        
        # Extract OPQ rotation matrix and PQ codebooks for differentiable optimization
        self.log(f'[TOKENIZER] Extracting OPQ rotation matrix...')
        self.opq_rotation = extract_opq_rotation_matrix(index)
        
        self.log(f'[TOKENIZER] Extracting PQ codebooks...')
        self.pq_codebooks = extract_pq_codebooks(
            index, 
            embedding_dim=sent_embs.shape[1],
            n_codebook=self.n_digit,
            codebook_size=self.codebook_size
        )
        
        # Original OPQ encoding to get semantic IDs
        ivf_index = faiss.downcast_index(index.index)
        invlists = faiss.extract_index_ivf(ivf_index).invlists
        ls = invlists.list_size(0)
        pq_codes = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
        pq_codes = pq_codes.reshape(-1, invlists.code_size)

        faiss_sem_ids = []
        n_bytes = pq_codes.shape[1]
        for u8code in pq_codes:
            bs = faiss.BitstringReader(faiss.swig_ptr(u8code), n_bytes)
            code = []
            for i in range(self.n_digit):
                code.append(bs.read(self.n_codebook_bits))
            faiss_sem_ids.append(code)
        pq_codes = np.array(faiss_sem_ids)

        item2sem_ids = {}
        for i in range(pq_codes.shape[0]):
            item = self.id2item[i + 1]
            item2sem_ids[item] = tuple(pq_codes[i].tolist())
        
        self.log(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')
        with open(sem_ids_path, 'w') as f:
            json.dump(item2sem_ids, f)

    def _sem_ids_to_tokens(self, item2sem_ids: dict) -> dict:
        """Converts semantic IDs to token IDs."""
        for item in item2sem_ids:
            tokens = list(item2sem_ids[item])
            for digit in range(self.n_digit):
                # Map each digit value d to token ID: codebook_size * digit + d + 1
                tokens[digit] += self.codebook_size * digit + 1
            item2sem_ids[item] = tuple(tokens)
        return item2sem_ids

    def _init_tokenizer(self, dataset: AbstractDataset):
        """Initialize tokenizer and load/generate semantic IDs."""
        sem_ids_path = os.path.join(
            dataset.cache_dir, 'processed',
            f'{os.path.basename(self.config["sent_emb_model"])}_{self.index_factory}.sem_ids'
        )

        if not os.path.exists(sem_ids_path):
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
            
            if self.config['sent_emb_pca'] > 0:
                self.log(f'[TOKENIZER] Applying PCA to sentence embeddings...')
                from sklearn.decomposition import PCA
                pca = PCA(n_components=self.config['sent_emb_pca'], whiten=True)
                sent_embs = pca.fit_transform(sent_embs)
            
            self.log(f'[TOKENIZER] Sentence embeddings shape: {sent_embs.shape}')

            training_item_mask = self._get_items_for_training(dataset)
            self._generate_semantic_id_opq(sent_embs, sem_ids_path, training_item_mask)

        self.log(f'[TOKENIZER] Loading semantic IDs from {sem_ids_path}...')
        item2sem_ids = json.load(open(sem_ids_path, 'r'))
        item2tokens = self._sem_ids_to_tokens(item2sem_ids)

        return item2tokens

    def _tokenize_first_n_items(self, item_seq: list) -> tuple:
        """Tokenizes the first n items in the given item_seq."""
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
        """Tokenizes the later items in the item sequence."""
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

    def tokenize_function(self, example: dict, split: str) -> dict:
        """Tokenizes an input example."""
        max_item_seq_len = self.config['max_item_seq_len']
        item_seq = example['item_seq'][0]
        
        if split == 'train':
            n_return_examples = max(len(item_seq) - max_item_seq_len, 1)

            input_ids, attention_mask, labels, seq_lens = self._tokenize_first_n_items(
                item_seq=item_seq[:min(len(item_seq), max_item_seq_len + 1)]
            )
            all_input_ids, all_attention_mask, all_labels, all_seq_lens = \
                [input_ids], [attention_mask], [labels], [seq_lens]

            for i in range(1, n_return_examples):
                cur_item_seq = item_seq[i:i+max_item_seq_len+1]
                input_ids, attention_mask, labels, seq_lens = self._tokenize_later_items(cur_item_seq)
                all_input_ids.append(input_ids)
                all_attention_mask.append(attention_mask)
                all_labels.append(labels)
                all_seq_lens.append(seq_lens)

            return {
                'input_ids': all_input_ids,
                'attention_mask': all_attention_mask,
                'labels': all_labels,
                'seq_lens': all_seq_lens,
            }
        else:
            input_ids, attention_mask, labels, seq_lens = self._tokenize_later_items(
                item_seq=item_seq[-(max_item_seq_len+1):],
                pad_labels=False
            )
            return {
                'input_ids': [input_ids],
                'attention_mask': [attention_mask],
                'labels': [labels[-1:]],
                'seq_lens': [seq_lens]
            }

    def tokenize(self, datasets: dict) -> dict:
        """Tokenizes the given datasets."""
        tokenized_datasets = {}
        for split in datasets:
            tokenized_datasets[split] = datasets[split].map(
                lambda t: self.tokenize_function(t, split),
                batched=True,
                batch_size=1,
                remove_columns=datasets[split].column_names,
                num_proc=self.config['num_proc'],
                desc=f'Tokenizing {split} set: '
            )

        for split in datasets:
            tokenized_datasets[split].set_format(type='torch')

        return tokenized_datasets