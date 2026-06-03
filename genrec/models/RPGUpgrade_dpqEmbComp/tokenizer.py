import os
import json
import numpy as np

from genrec.dataset import AbstractDataset
from genrec.models.RPG.tokenizer import RPGTokenizer


class RPGUpgrade_dpqEmbCompTokenizer(RPGTokenizer):
    """
    Extends RPGTokenizer to additionally expose:
        self.sent_embs    – np.ndarray (n_items, d), item_id-indexed (row 0 = zeros)
        self.opq_rotation – np.ndarray (d, d) or None if index unavailable
        self.pq_codebooks – np.ndarray (D, K, d/D) or None if index unavailable

    These are used by RPGUpgrade to warm-initialize the DPQ linear projection and
    Key/Value codebooks.
    """

    def _init_tokenizer(self, dataset: AbstractDataset):
        """
        Same as RPGTokenizer but also:
          1. Keeps the sentence embeddings in self.sent_embs
          2. Saves the FAISS index alongside .sem_ids for warm-init extraction
          3. Extracts OPQ rotation R and PQ codebooks K for DPQ warm-initialization
        """
        sem_ids_path = os.path.join(
            dataset.cache_dir, 'processed',
            f'{os.path.basename(self.config["sent_emb_model"])}_{self.index_factory}.sem_ids'
        )
        index_path = sem_ids_path.replace('.sem_ids', '.faiss')

        self.log(
            f'[TOKENIZER] ══════════════════════════════════════════════════════\n'
            f'[TOKENIZER]  RPGUpgrade _init_tokenizer() — overview\n'
            f'[TOKENIZER]    sem_ids cache : {sem_ids_path}\n'
            f'[TOKENIZER]    faiss cache   : {index_path}\n'
            f'[TOKENIZER]    index_factory : {self.index_factory}\n'
            f'[TOKENIZER] ══════════════════════════════════════════════════════'
        )

        # ------------------------------------------------------------------
        # 1. Load / encode sentence embeddings
        # ------------------------------------------------------------------
        self.log(f'[TOKENIZER] ======= 1. LOAD / ENCODE SENTENCE EMBEDDINGS =======')
        sent_emb_path = os.path.join(
            dataset.cache_dir, 'processed',
            f'{os.path.basename(self.config["sent_emb_model"])}.sent_emb'
        )
        if os.path.exists(sent_emb_path):
            self.log(f'[TOKENIZER] Sentence embeddings already exist in {sent_emb_path}')
            self.log(f'[TOKENIZER] Loading sentence embeddings from {sent_emb_path}...')
            sent_embs = np.fromfile(sent_emb_path, dtype=np.float32).reshape(
                -1, self.config['sent_emb_dim']
            )
        else:
            self.log(f'[TOKENIZER] Sentence embeddings not found — creating...')
            sent_embs = self._encode_sent_emb(dataset, sent_emb_path)

        if self.config['sent_emb_pca'] > 0:
            self.log(f'[TOKENIZER] Applying PCA to sentence embeddings...')
            self.log(f'[TOKENIZER] Embeddings shape before PCA: {sent_embs.shape}')
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.config['sent_emb_pca'], whiten=True)
            sent_embs = pca.fit_transform(sent_embs).astype(np.float32)
            self.log(f'[TOKENIZER] Embeddings shape after PCA: {sent_embs.shape}')
        else:
            self.log(f'[TOKENIZER] PCA disabled (sent_emb_pca=0); shape: {sent_embs.shape}')

        emb_dim = sent_embs.shape[1]

        # Build item_id-indexed table: row 0 = zero vector (padding)
        padded = np.zeros((sent_embs.shape[0] + 1, emb_dim), dtype=np.float32)
        padded[1:] = sent_embs
        self.sent_embs = padded  # shape: (n_items, emb_dim)
        self.log(
            f'[TOKENIZER] Built self.sent_embs table: shape={self.sent_embs.shape} '
            f'(row 0 = zero padding vector)'
        )

        # ------------------------------------------------------------------
        # 2. Build OPQ index if not already cached (also saves .faiss)
        # ------------------------------------------------------------------
        self.log(f'[TOKENIZER] ======= 2. BUILD / LOAD OPQ INDEX + SEMANTIC IDS =======')
        if not os.path.exists(sem_ids_path):
            self.log(f'[TOKENIZER] Semantic IDs mapping does not exist at {sem_ids_path}')
            self.log(f'[TOKENIZER] Will train OPQ, save .faiss and .sem_ids')
            self.log(f'[TOKENIZER] ================== 2a. GENERATE TRAINING ITEM MASK =============')
            training_item_mask = self._get_items_for_training(dataset)
            self.log(f'[TOKENIZER] ================== 2b. TRAIN OPQ + SAVE INDEX ===================')
            self._generate_semantic_id_opq_and_save_index(
                sent_embs, sem_ids_path, index_path, training_item_mask
            )
        elif not os.path.exists(index_path):
            self.log(
                f'[TOKENIZER] Semantic IDs exist at {sem_ids_path} but FAISS index '
                f'missing at {index_path}'
            )
            self.log(f'[TOKENIZER] Re-building FAISS index only (skip_sem_ids=True)...')
            training_item_mask = self._get_items_for_training(dataset)
            self._generate_semantic_id_opq_and_save_index(
                sent_embs, sem_ids_path, index_path, training_item_mask,
                skip_sem_ids=True
            )
        else:
            self.log(
                f'[TOKENIZER] Semantic IDs and FAISS index already cached — skipping OPQ training'
            )

        # ------------------------------------------------------------------
        # 3. Extract warm-init parameters from the saved FAISS index
        # ------------------------------------------------------------------
        self.log(f'[TOKENIZER] ======= 3. EXTRACT DPQ WARM-INIT PARAMS =======')
        if os.path.exists(index_path):
            self.log(f'[TOKENIZER] Loading FAISS index from {index_path} for OPQ param extraction...')
            self._extract_opq_params(index_path, emb_dim)
        else:
            self.log(
                f'[TOKENIZER] Warning: FAISS index not found at {index_path} — '
                f'DPQ module will use random initialization'
            )
            self.opq_rotation = None
            self.pq_codebooks = None

        # ------------------------------------------------------------------
        # 4. Load semantic IDs (same as parent)
        # ------------------------------------------------------------------
        self.log(f'[TOKENIZER] =============== 4. CREATING ITEM2TOKENS MAPPING... ==============')
        self.log(f'[TOKENIZER] Loading ITEM2SEM_IDS mapping from {sem_ids_path}...')
        item2sem_ids = json.load(open(sem_ids_path, 'r'))
        item2tokens = self._sem_ids_to_tokens(item2sem_ids)
        self.log(f'[TOKENIZER] Successfully created ITEM2TOKENS mapping ({len(item2tokens)} items)')
        return item2tokens

    # ------------------------------------------------------------------
    # Build OPQ index, save .faiss and optionally .sem_ids
    # ------------------------------------------------------------------
    def _generate_semantic_id_opq_and_save_index(
        self, sent_embs, sem_ids_path, index_path, train_mask, skip_sem_ids=False
    ):
        import faiss

        if self.config['opq_use_gpu']:
            res = faiss.StandardGpuResources()
            res.setTempMemory(1024 * 1024 * 512)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = self.n_digit >= 56

        faiss.omp_set_num_threads(self.config['faiss_omp_num_threads'])

        self.log(f'[TOKENIZER] Creating OPQ index with {self.index_factory}...')
        index = faiss.index_factory(
            sent_embs.shape[1],
            self.index_factory,
            faiss.METRIC_INNER_PRODUCT
        )

        self.log(
            f'[TOKENIZER] Training index on {train_mask.sum()} / {train_mask.shape[0]} items '
            f'(opq_use_gpu={self.config["opq_use_gpu"]})...'
        )
        if self.config['opq_use_gpu']:
            index = faiss.index_cpu_to_gpu(res, self.config['opq_gpu_id'], index, co)
        index.train(sent_embs[train_mask])
        index.add(sent_embs)
        if self.config['opq_use_gpu']:
            index = faiss.index_gpu_to_cpu(index)
        self.log(f'[TOKENIZER] Index trained and populated with {sent_embs.shape[0]} vectors')

        self.log(f'[TOKENIZER] Saving FAISS index to {index_path}...')
        faiss.write_index(index, index_path)

        if skip_sem_ids:
            self.log(
                f'[TOKENIZER] skip_sem_ids=True — keeping existing semantic IDs at {sem_ids_path}'
            )
        else:
            # Extract PQ codes and write .sem_ids (same logic as parent)
            ivf_index = faiss.downcast_index(index.index)
            invlists = faiss.extract_index_ivf(ivf_index).invlists
            ls = invlists.list_size(0)
            pq_codes = faiss.rev_swig_ptr(invlists.get_codes(0), ls * invlists.code_size)
            pq_codes = pq_codes.reshape(-1, invlists.code_size)

            faiss_sem_ids = []
            n_bytes = pq_codes.shape[1]
            for u8code in pq_codes:
                bs = faiss.BitstringReader(faiss.swig_ptr(u8code), n_bytes)
                code = [bs.read(self.n_codebook_bits) for _ in range(self.n_digit)]
                faiss_sem_ids.append(code)

            pq_codes_arr = np.array(faiss_sem_ids)
            item2sem_ids = {
                self.id2item[i + 1]: tuple(pq_codes_arr[i].tolist())
                for i in range(pq_codes_arr.shape[0])
            }

            self.log(
                f'[TOKENIZER] Extracted PQ codes for {pq_codes_arr.shape[0]} items; '
                f'saving ITEM2SEM_IDS to {sem_ids_path}...'
            )
            with open(sem_ids_path, 'w') as f:
                json.dump(item2sem_ids, f)
            self.log(f'[TOKENIZER] Semantic IDs saved successfully')

    # ------------------------------------------------------------------
    # Helper: extract R and K from saved FAISS index
    # ------------------------------------------------------------------
    def _extract_opq_params(self, index_path: str, emb_dim: int):
        """
        Extracts:
            self.opq_rotation – (d, d) float32 numpy array  (OPQ linear transform)
            self.pq_codebooks – (D, K, d/D) float32 numpy array  (PQ centroids)

        FAISS OPQ index structure:
            IndexPreTransform
              └── chain[0]: OPQMatrix (LinearTransform, stores A of shape d×d)
              └── index:    IndexIVFPQ (stores pq.centroids of shape D*K*(d/D))
        """
        import faiss

        index = faiss.read_index(index_path)
        self.log(f'[TOKENIZER] FAISS index loaded (type={type(index).__name__})')

        # --- Rotation / linear transform matrix ---
        # A is stored row-major as (d_out, d_in) = (d, d).
        # FAISS transform convention: y = x @ A^T  (same as nn.Linear weight).
        vt = faiss.downcast_VectorTransform(index.chain.at(0))
        R = faiss.vector_to_array(vt.A).reshape(emb_dim, emb_dim).copy()
        self.opq_rotation = R.astype(np.float32)  # (d, d)
        self.log(f'[TOKENIZER] Extracted opq_rotation: shape={self.opq_rotation.shape}')

        # --- PQ codebooks ---
        ivf_index = faiss.extract_index_ivf(faiss.downcast_index(index.index))
        ivf_pq = faiss.downcast_index(ivf_index)
        centroids = faiss.vector_to_array(ivf_pq.pq.centroids)
        sub_dim = emb_dim // self.n_digit
        self.pq_codebooks = centroids.reshape(
            self.n_digit, self.codebook_size, sub_dim
        ).copy().astype(np.float32)  # (D, K, d/D)
        self.log(
            f'[TOKENIZER] Extracted pq_codebooks: shape={self.pq_codebooks.shape} '
            f'(n_digit={self.n_digit}, codebook_size={self.codebook_size}, sub_dim={sub_dim})'
        )
        self.log(f'[TOKENIZER] DPQ warm-init ready — rotation + codebooks loaded from FAISS')
