import os
import json
import numpy as np

from genrec.dataset import AbstractDataset
from genrec.models.RPG.tokenizer import RPGTokenizer


class RPGUpgradeTokenizer(RPGTokenizer):
    """
    Extends RPGTokenizer to additionally expose:
        self.sent_embs      – np.ndarray (n_items, d), item_id-indexed (row 0 = zeros)
        self.opq_rotation   – np.ndarray (d, d) or None if index unavailable
        self.pq_codebooks   – np.ndarray (D, K, d/D) or None if index unavailable

    These are used by RPGUpgrade to warm-initialize the DPQ module.
    """

    def _init_tokenizer(self, dataset: AbstractDataset):
        """
        Same as RPGTokenizer but also:
          1. Keeps the sentence embeddings in self.sent_embs
          2. Saves the FAISS index alongside .sem_ids for warm-init extraction
          3. Extracts OPQ rotation R and PQ codebooks K from the saved index
        """
        sem_ids_path = os.path.join(
            dataset.cache_dir, 'processed',
            f'{os.path.basename(self.config["sent_emb_model"])}_{self.index_factory}.sem_ids'
        )
        index_path = sem_ids_path.replace('.sem_ids', '.faiss')

        # ------------------------------------------------------------------
        # 1. Load / encode sentence embeddings
        # ------------------------------------------------------------------
        sent_emb_path = os.path.join(
            dataset.cache_dir, 'processed',
            f'{os.path.basename(self.config["sent_emb_model"])}.sent_emb'
        )
        if os.path.exists(sent_emb_path):
            self.log('[TOKENIZER] Loading sentence embeddings...')
            sent_embs = np.fromfile(sent_emb_path, dtype=np.float32).reshape(
                -1, self.config['sent_emb_dim']
            )
        else:
            self.log('[TOKENIZER] Encoding sentence embeddings...')
            sent_embs = self._encode_sent_emb(dataset, sent_emb_path)

        if self.config['sent_emb_pca'] > 0:
            self.log('[TOKENIZER] Applying PCA...')
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.config['sent_emb_pca'], whiten=True)
            sent_embs = pca.fit_transform(sent_embs).astype(np.float32)

        emb_dim = sent_embs.shape[1]

        # Build item_id-indexed table: row 0 = zero vector (padding)
        padded = np.zeros((sent_embs.shape[0] + 1, emb_dim), dtype=np.float32)
        padded[1:] = sent_embs
        self.sent_embs = padded  # shape: (n_items, emb_dim)

        # ------------------------------------------------------------------
        # 2. Build OPQ index if not already cached (also saves .faiss)
        # ------------------------------------------------------------------
        if not os.path.exists(sem_ids_path):
            self.log(f'[TOKENIZER] Embeddings shape: {sent_embs.shape}')
            training_item_mask = self._get_items_for_training(dataset)
            self._generate_semantic_id_opq_and_save_index(
                sent_embs, sem_ids_path, index_path, training_item_mask
            )
        elif not os.path.exists(index_path):
            # sem_ids exist (created by RPGTokenizer) but .faiss was not saved – re-build
            self.log('[TOKENIZER] FAISS index not found alongside .sem_ids; re-building...')
            training_item_mask = self._get_items_for_training(dataset)
            self._generate_semantic_id_opq_and_save_index(
                sent_embs, sem_ids_path, index_path, training_item_mask,
                skip_sem_ids=True  # don't overwrite the existing .sem_ids
            )

        # ------------------------------------------------------------------
        # 3. Extract warm-init parameters from the saved FAISS index
        # ------------------------------------------------------------------
        if os.path.exists(index_path):
            self.log('[TOKENIZER] Extracting OPQ params from FAISS index...')
            self._extract_opq_params(index_path, emb_dim)
        else:
            self.log('[TOKENIZER] Warning: FAISS index not found – DPQ will use random init.')
            self.opq_rotation = None
            self.pq_codebooks = None

        # ------------------------------------------------------------------
        # 4. Load semantic IDs (same as parent)
        # ------------------------------------------------------------------
        self.log('[TOKENIZER] Loading semantic IDs...')
        item2sem_ids = json.load(open(sem_ids_path, 'r'))
        item2tokens = self._sem_ids_to_tokens(item2sem_ids)
        return item2tokens

    # ------------------------------------------------------------------
    # Helper: build OPQ index, save .faiss and optionally .sem_ids
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

        index = faiss.index_factory(
            sent_embs.shape[1],
            self.index_factory,
            faiss.METRIC_INNER_PRODUCT
        )

        self.log('[TOKENIZER] Training index...')
        if self.config['opq_use_gpu']:
            index = faiss.index_cpu_to_gpu(res, self.config['opq_gpu_id'], index, co)
        index.train(sent_embs[train_mask])
        index.add(sent_embs)
        if self.config['opq_use_gpu']:
            index = faiss.index_gpu_to_cpu(index)

        # Save FAISS index for DPQ warm-init
        self.log(f'[TOKENIZER] Saving FAISS index to {index_path}...')
        faiss.write_index(index, index_path)

        if not skip_sem_ids:
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

            self.log(f'[TOKENIZER] Saving semantic IDs to {sem_ids_path}...')
            with open(sem_ids_path, 'w') as f:
                json.dump(item2sem_ids, f)

    # ------------------------------------------------------------------
    # Helper: extract R and K from saved FAISS index
    # ------------------------------------------------------------------
    def _extract_opq_params(self, index_path: str, emb_dim: int):
        """
        Extracts:
            self.opq_rotation  – (d, d) float32 numpy array  (the OPQ rotation matrix A)
            self.pq_codebooks  – (D, K, d/D) float32 numpy array  (PQ centroids)

        FAISS OPQ index structure:
            IndexPreTransform
              └── chain[0]: OPQMatrix (LinearTransform, stores A of shape d×d)
              └── index:    IndexIVFPQ (stores pq.centroids of shape D*K*(d/D))
        """
        import faiss

        index = faiss.read_index(index_path)

        # --- Rotation matrix ---
        vt = faiss.downcast_VectorTransform(index.chain.at(0))
        # A is stored row-major as (d_out, d_in) = (d, d)
        # FAISS transform: y = x @ A^T  →  same convention as nn.Linear weight
        R = faiss.vector_to_array(vt.A).reshape(emb_dim, emb_dim).copy()
        self.opq_rotation = R.astype(np.float32)  # (d, d)

        # --- PQ codebooks ---
        ivf_pq = faiss.extract_index_ivf(faiss.downcast_index(index.index))
        centroids = faiss.vector_to_array(ivf_pq.pq.centroids)
        sub_dim = emb_dim // self.n_digit
        self.pq_codebooks = centroids.reshape(
            self.n_digit, self.codebook_size, sub_dim
        ).copy().astype(np.float32)  # (D, K, d/D)
