# This document explain the idea of PQ (Product Quantization) and the basic idea of making it differentiable via the Gumbel-Softmax trick
# The work I will be doing to upgrade the model RPG is not 100% idential to the stuff written in this file, however this file serve as a reference
---

## 1. Notations and Vector Spaces

### Dimensional & Hyperparameters:
* $d$: Dimension of the original continuous item embedding space.
* $D$: Number of subspaces (corresponding to the number of code digits/tokens per item).
* $d/D$: Dimension of each sliced sub-vector/subspace.
* $K$: Number of cluster centroids (codebook size) per subspace.
* $n$: Batch size (number of items in the current training iteration).

### Matrices and Variables:
* $\mathbf{Q} \in \mathbb{R}^{n \times d}$: The input query matrix containing raw continuous embeddings of the items in the batch.
* $\mathbf{Q}_i^{(j)} \in \mathbb{R}^{d/D}$: The $j$-th sub-vector (subspace segment) of the $i$-th item.
* $\mathbf{K} \in \mathbb{R}^{K \times d}$: The Key matrix containing the learnable cluster centroids used for distance/similarity calculations. It is partitioned into $D$ sub-codebooks: $\mathbf{K}^{(j)} \in \mathbb{R}^{K \times d/D}$.
* $\mathbf{V} \in \mathbb{R}^{K \times d}$: The Value matrix containing the embedding vectors used to reconstruct the continuous space. It is partitioned into $D$ sub-codebooks: $\mathbf{V}^{(j)} \in \mathbb{R}^{K \times d/D}$.
* $\mathbf{C} \in \{1, \dots, K\}^{n \times D}$: The discrete code matrix (Codebook indices). The element $\mathbf{C}_i^{(j)}$ represents the discrete token index (from $1$ to $K$) assigned to the $i$-th item at the $j$-th slot.
* $\mathbf{H} \in \mathbb{R}^{n \times d}$: The final reconstructed continuous embedding matrix after passing through the DPQ bottleneck.

---

## 2. Standard Product Quantization Mechanism (Non-Differentiable)

The general DPQ function acts as a mapping between two continuous spaces, $\mathcal{T} : \mathbb{R}^d \to \mathbb{R}^d$, operating through a discrete bottleneck $\{1, \dots, K\}^D$ via two major functions:
$$\mathcal{T}(\cdot) = \rho \circ \phi(\cdot)$$

1. **Discretization Function $\phi(\cdot)$:** Maps a continuous sub-vector into a discrete index by identifying the nearest centroid based on a distance metric $\text{dist}(\cdot, \cdot)$:
   $$\mathbf{C}_i^{(j)} = \arg \min_k \text{dist}\left(\mathbf{Q}_i^{(j)}, \mathbf{K}_k^{(j)}\right)$$

2. **Reverse-Discretization Function $\rho(\cdot)$:** Maps the discrete indices back into a continuous embedding vector by looking up and concatenating vectors from the Value matrix $\mathbf{V}$:
   $$\mathbf{H}_i = \rho(\mathbf{C}_i) = \left[\mathbf{V}_{\mathbf{C}_i^{(1)}}^{(1)}, \dots, \mathbf{V}_{\mathbf{C}_i^{(j)}}^{(j)}, \dots, \mathbf{V}_{\mathbf{C}_i^{(D)}}^{(D)}\right]$$

Because the $\arg \min$ (or $\arg \max$) operation is a step function with a gradient of zero everywhere, backpropagation is blocked.

---

## 3. Softmax-based Relaxation with Gumbel Noise

To enable gradient flow, the discrete selection is relaxed using a weighted Softmax function. Additionally, i.i.d. Gumbel noise is introduced to encourage codebook exploration and prevent codebook collapse.

### Step 1: Compute Similarity Logits ($\ell$)
For the $i$-th item in the $j$-th subspace, we compute the similarity logits against each centroid $k$ using the dot product (or negative Euclidean distance):
$$\ell_{i,j,k} = \text{sim}\left(\mathbf{Q}_i^{(j)}, \mathbf{K}_k^{(j)}\right) = \langle \mathbf{Q}_i^{(j)}, \mathbf{K}_k^{(j)} \rangle$$

### Step 2: Inject Gumbel Noise and Compute Soft Probabilities ($\tilde{\mathbf{C}}$)
We sample i.i.d. noise $g_{i,j,k} \sim \text{Gumbel}(0, 1)$ and apply the Softmax function controlled by a temperature hyperparameter $\tau$:
$$\tilde{\mathbf{C}}_{i,k}^{(j)} = \frac{\exp\left((\ell_{i,j,k} + g_{i,j,k})/\tau\right)}{\sum_{k'=1}^K \exp\left((\ell_{i,j,k'} + g_{i,j,k'})/\tau\right)}$$

Here, $\tilde{\mathbf{C}}_i^{(j)} \in \Delta^K$ represents a **soft one-hot probability vector** instead of a single hard integer index.

---

## 4. Straight-Through Estimator (STE) Mechanism

To bridge the gap between training and inference, we employ the Straight-Through Estimator trick. This ensures that the **Forward Pass outputs discrete hard codes**, while the **Backward Pass propagates gradients smoothly via soft probabilities**.

### Formulating Subspace Output Vectors:
* **Soft Representation ($\mathbf{H}_{i, soft}^{(j)}$):** Computed as the expected value (weighted average) of the Value matrix vectors based on the soft distribution:
  $$\mathbf{H}_{i, soft}^{(j)} = \sum_{k=1}^K \tilde{\mathbf{C}}_{i,k}^{(j)} \mathbf{V}_k^{(j)} = \tilde{\mathbf{C}}_i^{(j)} \mathbf{V}^{(j)}$$

* **Hard Representation ($\mathbf{H}_{i, hard}^{(j)}$):** Sampled discretely via $\arg \max$ (equivalent to setting $\tau \to 0$):
  $$\mathbf{C}_i^{(j)} = \arg \max_k \left(\ell_{i,j,k} + g_{i,j,k}\right)$$
  $$\mathbf{H}_{i, hard}^{(j)} = \mathbf{V}_{\mathbf{C}_i^{(j)}}^{(j)}$$

### STE Formula using the Stop-Gradient Operator ($\text{sg}$):
To force PyTorch/Copilot to execute this behavior, the final output vector for the $j$-th subspace of the $i$-th item is constructed as:
$$\mathbf{H}_i^{(j)} = \mathbf{H}_{i, hard}^{(j)} + \mathbf{H}_{i, soft}^{(j)} - \text{sg}\left(\mathbf{H}_{i, soft}^{(j)}\right)$$

* **Forward Pass Execution:** Since $\mathbf{H}_{i, soft}^{(j)} - \text{sg}\left(\mathbf{H}_{i, soft}^{(j)}\right) = 0$, the mathematical value fed into the downstream Transformer layers is exactly the discrete hard representation $\mathbf{H}_{i, hard}^{(j)}$.
* **Backward Pass Execution:** The stop-gradient operator $\text{sg}(\cdot)$ zero out the gradients of its inner term and the non-differentiable hard term. Consequently, the gradient operator $\nabla$ bypasses them and flows directly through the continuous $\mathbf{H}_{i, soft}^{(j)}$ path. This updates both the Key matrix $\mathbf{K}$ and Value matrix $\mathbf{V}$ end-to-end.