---
title: "SpectralNet: Teaching a Neural Network to Cluster Like an Eigenvector Solver"
author: Ayush Jangid
date: 2026-07-31
categories: [Paper Implementations]
tags: [Machine Learning, Research, Roadmap, PaperImplementations, SpectralNet]
---


---
> **TL;DR**
> - Spectral clustering finds structure by embedding data into the eigenspace of a graph Laplacian — but it costs O(n³) and can't generalize to new points.
> - SpectralNet trains a neural network to *approximate* those eigenvectors on mini-batches, making spectral clustering scalable and out-of-sample extensible.
> - The key trick is a differentiable Cholesky orthogonalization layer that enforces the eigenvector orthonormality constraint inside the forward pass, letting standard gradient descent handle a constrained optimization problem.

---

## Introduction

Spectral clustering is one of those techniques that feels almost magical when you first encounter it. Build a graph of your data, compute its Laplacian, find a handful of eigenvectors — and suddenly the cluster structure you couldn't see in raw pixel space jumps out cleanly. It works on rings, moons, concentric circles, and all the non-convex shapes that k-means has no hope on.

There is just one problem: the eigenvector computation scales as O(n³). For MNIST's 70,000 images, that means constructing and decomposing a 70,000 × 70,000 matrix. It simply isn't feasible. And even if you could compute it, the eigenvectors give you an embedding *only for points you've seen* — a new test image has no natural home in that eigenspace. Every time you add a data point, you'd have to rerun the whole decomposition.

The SpectralNet paper (Shaham et al., 2018) asks a beautifully simple question: *what if we trained a neural network to do what the eigenvector solver does — but stochastically, on mini-batches, so it scales — and because it's a network, it automatically generalizes?*

The surprising result: on the Reuters dataset, SpectralNet achieved state-of-the-art clustering accuracy at the time, while being applicable to datasets far too large for classical spectral methods. On MNIST, you can expect 80–92% clustering accuracy in a fully unsupervised setting.

---

## Plain-English Intuition

Before any math: what is SpectralNet *actually doing*?

Think of it like a radio tower assignment problem. You have thousands of neighborhoods (data points), and you want to assign each one to a city (cluster). The natural rule is: neighborhoods that can easily reach each other (similar data points) should belong to the same city. Spectral clustering formalizes this by building a "connectivity graph" between neighborhoods, then finding the natural "basins" inthat graph using eigenvectors.

The eigenvectors of the graph Laplacian are — by construction — the smoothest possible functions over the graph. The first k eigenvectors carve the graph into k regions while minimizing the amount of "cutting" across high-connectivity edges. Running k-means in that eigenspace is then trivially easy.

SpectralNet replaces the eigenvector solver with a neural network. The network is trained to output a k-dimensional embedding for each data point such that:

1. **Similar points (high graph affinity) get similar embeddings** — the spectral loss penalizes large distances between connected points.
2. **The outputs stay orthonormal** — a Cholesky layer inside the forward pass enforces this as a hard constraint, preventing the trivial "map everything to zero" collapse.

Because it's a neural network, you train it on mini-batches (scalability) and then just do a forward pass for new points (generalization). The rest is engineering to make this clean idea actually work.

---

## Theory and Math

### The Spectral Clustering Objective

Classical spectral clustering solves:

```
Y* = argmin_Y  trace(Y^T L Y)   subject to   Y^T Y = I
```

Let's unpack each symbol:
- **Y ∈ ℝ^{n×k}**: the embedding matrix. Each row is one data point's k-dimensional spectral embedding. n = number of data points, k = number of clusters.
- **L = D − W**: the graph Laplacian. W is the affinity (similarity) matrix, where W_ij is large when points i and j are similar. D is the diagonal degree matrix, with D_ii = Σ_j W_ij (total affinity of point i to all others).
- **trace(Y^T L Y)**: the Rayleigh quotient. It equals Σ_{i,j} W_ij · ‖y_i − y_j‖², which is the sum of squared distances between all pairs, *weighted by how similar they are*. Minimizing this pulls similar points together in embedding space.
- **Y^T Y = I**: the orthonormality constraint. Without it, the optimizer would simply map every point to zero (trivial minimum). This constraint forces the embedding to span a k-dimensional space non-degenerately.

The solution Y* consists of the k eigenvectors of L corresponding to its k smallest eigenvalues. That's classical spectral clustering.

### Approximating It with Mini-Batches

SpectralNet approximates this objective on a mini-batch of m points. For a batch, you have:
- **Y_batch ∈ ℝ^{m×k}**: network output for m inputs
- **W_batch ∈ ℝ^{m×m}**: affinity matrix built *only from this batch*

The loss becomes:

```
L_spectral = (1/m) · trace(Y_orth^T L_batch Y_orth)
           = (1/m) · Σ_{i,j} W_ij · ‖y_i − y_j‖²
```

- **(1/m)**: normalizes the loss so its scale doesn't depend on batch size
- **W_ij**: a large value here means points i and j are similar — the loss strongly penalizes them being far apart in embedding space
- **‖y_i − y_j‖²**: squared Euclidean distance between spectral embeddings of points i and j

The second form is the most intuitive: it's a weighted sum of embedding distances, where the weights are how similar the points are. Minimize this → similar points cluster together.

### The Affinity Matrix: Self-Tuning Gaussian Kernel

To build W on a mini-batch, we use a Gaussian kernel with a *per-point* bandwidth — this is the "self-tuning" kernel from Zelnik-Manor & Perona (2004):

```
σ_i = mean distance from x_i to its k nearest neighbors (k=10)

W_ij = exp(−‖x_i − x_j‖² / (σ_i · σ_j))   if j is a k-NN of i
W_ij = 0                                      otherwise
```

- **σ_i**: local scale parameter for point i. Dense regions get small σ, sparse regions get large σ. This makes the kernel adaptive to local density.
- **σ_i · σ_j**: geometric mean of the two bandwidths — symmetric by construction.
- **k-NN masking**: we zero out W_ij for non-neighbors before applying the kernel. This sparsity is both computationally useful and conceptually right: only local connections should determine cluster structure.

### Contrastive Loss for the Siamese Network

The Siamese network is trained with contrastive loss. For a pair (x_i, x_j) with label y (0 = same class, 1 = different class):

```
d = ‖g_φ(x_i) − g_φ(x_j)‖₂

L_contrastive = (1−y) · d²  +  y · max(0, margin − d)²
```

- **d**: L2 distance between the two embeddings
- **(1−y) · d²**: for positive pairs (y=0), penalize large distance — pull them together
- **y · max(0, margin − d)²**: for negative pairs (y=1), penalize being *too close* — push them apart, but only up to the margin (margin = 2.0). Beyond that margin, negative pairs are fine where they are.

This loss shapes the embedding space so that intra-class distances are small and inter-class distances are at least `margin`.

---

## Architecture Walkthrough

The full pipeline has four active components at training time:

```
 Raw MNIST Images (784-dim flat vectors)
         │
         ├────────────────────────────────┐
         │ [for affinity only]            │ [SpectralNet input]
         ▼                               │
 ┌─────────────────┐                     │
 │  Siamese MLP    │ g_φ (frozen)        │
 │  784→1024→1024  │                     │
 │  →512→10        │                     │
 └────────┬────────┘                     │
          │ Embeddings ℝ^{m×10}          │
          ▼                              │
 ┌─────────────────────────────┐         │
 │  k-NN Graph + Self-tuning   │         │
 │  Gaussian Kernel            │         │
 │  W ∈ ℝ^{m×m}               │         │
 └────────┬────────────────────┘         │
          │                              ▼
          │                    ┌──────────────────┐
          │                    │  SpectralNet MLP  │
          │                    │  784→1024→1024    │
          │                    │  →512→10          │
          │                    │  Y_raw ∈ ℝ^{m×10}│
          │                    └────────┬──────────┘
          │                             │
          │                             ▼
          │                    ┌──────────────────────────┐
          │                    │  Cholesky Orthogonalizer  │
          │                    │  Y_orth = Y(L^T)^{-1}    │
          │                    │  Y_orth ∈ ℝ^{m×10}      │
          │                    └────────┬─────────────────┘
          │                             │
          └──────────────┬──────────────┘
                         ▼
              ┌────────────────────┐
              │   Spectral Loss    │
              │  (1/m)·Σ W_ij·    │
              │  ‖y_i − y_j‖²     │
              └────────────────────┘
                         │
                    backprop into SpectralNet only
                    (Siamese is frozen)
```

### Component 1: Siamese MLP

**In:** Two images x_i, x_j ∈ ℝ^{784} (flat MNIST pixels)  
**Out:** Two embeddings z_i, z_j ∈ ℝ^{10}  
**Why this shape:** The architecture `[784→1024→1024→512→10]` mirrors SpectralNet's own depth and capacity. d_embed=10 matches the number of clusters — keeping the affinity space compact and interpretable.

The Siamese is only used for **building W**. After training, it's frozen and acts as a feature extractor for the affinity graph. SpectralNet itself still receives raw pixels as input.

### Component 2: k-NN Affinity Matrix

**In:** Siamese embeddings ℝ^{m×10}  
**Out:** Sparse symmetric affinity matrix W ∈ ℝ^{m×m}  
**Why k-NN masking:** Without sparsification, the affinity matrix would connect every point to every other — creating a graph with no useful structure. k=10 neighbors preserves local topology while discarding long-range noise.

### Component 3: SpectralNet MLP

**In:** Raw pixels ℝ^{m×784}  
**Out (before orthogonalization):** Y_raw ∈ ℝ^{m×10}, linear activations  
**Key design choice:** The final layer has **no activation function**. This is intentional — the output must be unconstrained for the Cholesky layer to work correctly. Adding a sigmoid or tanh here would destroy the orthogonalization.

### Component 4: Cholesky Orthogonalization

**In:** Y_raw ∈ ℝ^{m×k}  
**Out:** Y_orth ∈ ℝ^{m×k} with Y_orth^T Y_orth = I  
**Why Cholesky:** We need to whiten Y_raw — transform it so its columns are orthonormal. Cholesky decomposition gives us the "square root" of Y^T Y, which we can invert analytically. The operation is fully differentiable (PyTorch autograd handles `torch.linalg.cholesky`), so gradients flow through it without any special treatment.

---

## Code Deep-Dives

### The Siamese Network: Architecture and Contrastive Loss

The Siamese architecture is refreshingly minimal. Both "arms" share the same weights — there is only one set of parameters:

```python
class SiameseTwin(nn.Module):
    def __init__(self, input_dim=784, d_embed=10):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 1024),      nn.ReLU(),
            nn.Linear(1024, 512),       nn.ReLU(),
            nn.Linear(512, d_embed)     # no activation — raw L2 distances need linear output
        )

    def contrastive_loss(self, x1, x2, y, margin=2.0):
        # x1, x2: embeddings of each pair; y=0 same class, y=1 different class
        dist = torch.norm(x1 - x2, p=2, dim=1)  # L2 distance per pair → (B,)
        
        # positive term: same-class pairs, drive distance to 0
        # negative term: different-class pairs, drive distance beyond margin
        return torch.mean(
            (1 - y) * dist**2 + y * torch.clamp(margin - dist, min=0)**2
        )
```

One subtlety: `torch.clamp(margin - dist, min=0)` — when a negative pair is already farther than `margin`, the gradient is exactly zero. The optimizer ignores those pairs. This is the hinge behavior of contrastive loss, and it prevents the network from wasting capacity pushing already-separated negatives even farther apart.

### Pair Sampling: The Smart Way

The training script doesn't pre-generate pairs. Instead, it mines them on-the-fly from each mini-batch:

```python
# (B, B) boolean: True where two samples have the same label
same_class = y[:, None] == y[None, :]
same_class.fill_diagonal_(False)   # a point is not its own pair

# keep only upper triangle → avoid counting (i,j) and (j,i) as separate pairs
upper = torch.triu(torch.ones_like(same_class, dtype=torch.bool), diagonal=1)

positive_mask = same_class & upper
negative_mask = (~same_class) & upper

# sample a balanced subset (capped at B//2 each to avoid memory blow-up)
pos_idx = torch.randperm(len(positive_pairs[0]))[:num_pos]
neg_idx = torch.randperm(len(negative_pairs[0]))[:num_neg]
```

This is elegant: a batch of 256 images contains up to ~3000 valid pairs, and you sample a balanced subset each step. No separate pair dataset needed. The `triu` trick avoids duplicate pairs without any set operations.

### The Heart of SpectralNet: Spectral Loss

This is where all the pieces come together. The `spectral_loss` method does affinity construction, orthogonalization, and loss computation in one shot:

```python
def spectral_loss(self, x_inp, op):
    # op: raw SpectralNet output, shape (B, k), NOT yet orthogonalized
    
    y = x_inp
    if self.siamese is not None:
        with torch.no_grad():   # Siamese is frozen — no gradient needed here
            y = self.siamese(y) # (B, 10) — Siamese embedding space for affinity
    
    # --- Build affinity matrix in Siamese space ---
    cdist = self.pairwise_dist(y)   # (B, B) squared distances
    
    # Find k=10 nearest neighbors per point (excluding self: start from index 1)
    sorted_indices = torch.argsort(cdist, dim=-1)[:, 1:self.affinity_matrix_clusters+1]
    mask = torch.zeros_like(cdist).scatter_(1, sorted_indices, 1.0)  # k-NN binary mask
    
    # Self-tuning bandwidth: σ_i = mean distance to k-NN (in sqrt-distance space)
    mean_dist = torch.sum(torch.sqrt(cdist) * mask, dim=-1) / self.affinity_matrix_clusters
    sigma = (mean_dist[:, None]) @ (mean_dist[None, :]) + 1e-8  # outer product → (B, B)
    
    W = torch.exp(-cdist / (2 * sigma)) * mask  # Gaussian kernel, k-NN masked
    W = (W + W.T) / 2                           # symmetrize
    
    # --- Orthogonalize SpectralNet output ---
    spec_op = self.orthogonlize(op)  # (B, k)
    
    # --- Spectral loss: (1/m) * Σ W_ij * ||y_i - y_j||² ---
    cdist_spec = self.pairwise_dist(spec_op)   # distance in EMBEDDING space
    return (1 / spec_op.shape[0]) * torch.sum(W * cdist_spec)
```

Note the two separate pairwise distance calls: `cdist` is computed in **Siamese space** (for W), and `cdist_spec` is computed in **spectral embedding space** (for the loss). They are intentionally different — W encodes the desired cluster structure, while cdist_spec measures how well the current embedding respects it.

### Cholesky Orthogonalization: Differentiable Whitening

```python
def orthogonlize(self, op):
    # op: (B, k) raw network output
    
    # Gram matrix: S = Y^T Y, shape (k, k)
    # Add ε*I for numerical stability — prevents Cholesky from failing on near-singular S
    S = op.T @ op + self.eps * torch.eye(self.output_clusters, device=op.device, dtype=op.dtype)
    
    # Cholesky: S = L L^T, L is lower-triangular
    L = torch.linalg.cholesky(S)  # (k, k)
    
    # Solve L^T @ op_orth^T = op^T  →  op_orth = op @ (L^T)^{-1}
    # Using solve_triangular is more stable than explicitly inverting L
    op_orth = torch.linalg.solve_triangular(L, op.T, upper=False).T  # (B, k)
    
    return op_orth
    # Verify: op_orth^T @ op_orth = (L^{-1} Y^T)(Y L^{-T}) = L^{-1}(Y^T Y)L^{-T}
    #                              = L^{-1}(LL^T)L^{-T} = I  ✓
```

This is the most mathematically dense part of the codebase, but the geometric meaning is simple: Cholesky orthogonalization is just whitening — it rescales and rotates the columns of Y so they become orthonormal. The important thing is that `torch.linalg.cholesky` and `torch.linalg.solve_triangular` are both differentiable, so backprop flows through them automatically.

### Evaluation: k-Means + Hungarian Algorithm

After training, we encode the full dataset and evaluate with clustering accuracy:

```python
def kmeans_eval(model, loader, device):
    embeddings, labels = [], []
    for batch in loader:
        x, y = batch
        x = x.reshape(x.shape[0], -1).to(device)
        with torch.no_grad():
            op = model(x)              # (B, k)
            op = model.orthogonlize(op)  # (B, k), orthogonalized
        embeddings.append(op)
        labels.append(y)
    
    embeddings = torch.vstack(embeddings).cpu().numpy()  # (N, k)
    labels = torch.cat(labels).numpy()
    
    # n_init=100: run k-means 100 times from random initializations, keep best
    # This is important — k-means is sensitive to initialization
    kmeans = KMeans(n_clusters=10, n_init=100, random_state=0).fit(embeddings)
    preds = kmeans.labels_
    
    # Hungarian algorithm: find the best permutation of cluster labels → true labels
    cm = confusion_matrix(labels, preds)
    row_ind, col_ind = linear_sum_assignment(-cm)  # maximize trace → minimize negative
    acc = cm[row_ind, col_ind].sum() / labels.shape[0]
    return acc
```

The Hungarian algorithm is what makes the accuracy metric meaningful. Cluster IDs are arbitrary — what we care about is whether cluster0 perfectly aligns with true label "3", cluster 1 aligns with "7", etc. `linear_sum_assignment` finds the optimal assignment.

---

## Implementation Gotchas

**1. SpectralNet takes raw pixels — not Siamese embeddings.**

This is the most confusing part. The Siamese output is *only* used to build the affinity matrix W. SpectralNet itself always receives raw 784-dim pixel vectors as input. I originally passed Siamese embeddings into SpectralNet, got loss values that looked reasonable, and only caught the error when my clustering accuracy stagnated around 60%. The fix is one line — but finding it cost hours.

**2. Cholesky must be inside the computational graph.**

If you apply Cholesky *after* calling `.detach()` on the network output, gradients stop there. The orthogonalization is then just a post-processing step, and SpectralNet has no learning signal about the orthogonality constraint. The loss will still decrease, but what you're optimizing is no longer the spectral objective — you're minimizing an unconstrained version, which can degenerate.

**3. The `ε` in the Gram matrix matters more than you'd think.**

At the start of training, Y_raw is near-random. Its columns can be nearly linearly dependent, making S = Y^T Y nearly singular. With `eps=0`, Cholesky fails outright with a "matrix not positive definite" error. With `eps=1e-4` it's stable. I initially used `eps=1e-6` andgot NaN losses in the first 5 epochs — bumping to `eps=1e-4` fixed it.

**4. Pair sampling balance matters for Siamese training.**

MNIST has 10 classes, so random pairs are ~90% negative (different class). Training with unbalanced pairs made my Siamese network excellent at pushing negatives apart, but almost useless at pulling positives together. The `triu + balanced sampling` approach in the code (equal positive/negative pairs per batch) is essential. With unbalanced pairs, the Siamese-built affinity matrix was noisy enough to hurtSpectralNet's final clustering accuracy by ~8–10%.

**5. Orthogonalize at inference using the full dataset, not per-batch.**

During training, Cholesky is applied per mini-batch of 1024 points. At inference, if you naively do the same on 256-point batches, the orthogonalization is relative to each batch, not the global dataset. The batch-by-batch Cholesky at inference gives embeddings that are incoherent across batches — k-means on them gives near-random results. The fix: collect all embeddings without orthogonalizing, then apply one global Cholesky on the full N×k matrix.

**6. Self-tuning σ vs. fixed σ: not optional.**

I initially tried a fixed `σ=1.0`. The Gaussian kernel is extremely sensitive to the absolute scale of distances in the Siamese space, which varies unpredictably across datasets and training stages. With fixed σ, the affinity matrix was either too dense (everything connected to everything) or too sparse (nothing connected to anything) depending on the batch. Self-tuning σ adapts to local geometry and is robust across all training phases.

---

## Results and Validation

On MNIST (60k train / 10k test), my implementation achieved:

| Split      | Clustering Accuracy (ACC) |
|------------|--------------------------|
| Training   | ~0.86                    |
| Validation | ~0.84                    |

The paper reports up to ~0.92 on MNIST. The gap is likely due to:
- Using n_init=100 vs. potentially more in the paper
- Mini-batch size: I used 256 (the DataLoader default) vs. the paper's recommended 1024. Larger batches give better approximation of the global Laplacian.
- Siamese training for 10 epochs (paper used up to 15).

**Training dynamics to expect:**

Siamese loss starts around 0.8–1.2 and converges to 0.05–0.15. A key diagnostic: `pos_dist` (mean embedding distance for same-class pairs) should drop to near zero, while `neg_dist` (different-class pairs) should plateau above the margin threshold. If both converge to the same value, the Siamese isn't learning meaningful separations.

SpectralNet spectral loss starts at 0.3–0.8 and should converge to 0.05–0.2. If it drops below 0.01 in the first few epochs and stays there, something is wrong — likely the Cholesky is collapsing (check eps), or the affinity matrix W is all-zeros.

---

## Reflection and Takeaways

The most elegant thing about SpectralNet is that it converts a *constrained* optimization problem (minimize Rayleigh quotient subject to orthonormality) into an *unconstrained* one (train a network with a differentiable constraint layer). The constraint isn't relaxed or penalized — it's exactly enforced, every forward pass, by a two-line Cholesky computation. That design choice is what makes the whole thing work.

The Siamese network's role surprised me. I expected the spectral loss alone to be the hard part. But in practice, the quality of the affinity graph W is the dominant factor in final clustering accuracy. A good Siamese embedding makes W capture true semantic similarity; apoor one makes W noisy and the spectral embedding can't recover. The paper proves a theoretical lower bound on SpectralNet size (via VCdimension), but empirically, the bottleneck is almost always the affinity graph, not the network capacity.

**Two open questions I still think about:**

1. **Can the Siamese be replaced by a self-supervised pretraining step?** The Siamese requires label information for pair construction.For truly unsupervised settings (no labels at all), you'd need to define "similar" without supervision. SimCLR-style augmentation pairsmight work here — but would the resulting affinity matrix capture the same cluster structure? It's not obvious.

2. **Is the mini-batch Laplacian approximation theoretically sound?** The paper claims stochastic optimization converges to the global spectral embedding, but the proof is informal. Each mini-batch computes a completely different affinity matrix on a random subset — the loss function changes every step. This is closer to online non-stationary optimization than standard SGD. Under what conditions does this actually converge to the global eigenvectors, and how fast?

---

## Further Reading

**Papers that led to SpectralNet:**
- **Ng, Jordan & Weiss (2002)** — "On Spectral Clustering: Analysis and an Algorithm." The original spectral clustering paper. Read this before SpectralNet; the intuition maps directly.
- **Zelnik-Manor & Perona (2004)** — "Self-Tuning Spectral Clustering." The source of the self-tuning Gaussian kernel used for σ in SpectralNet's affinity matrix.
- **Hadsell, Chopra & LeCun (2006)** — "Dimensionality Reduction by Learning an Invariant Mapping." The contrastive loss paper — the direct ancestor of the Siamese training used here.

**Papers that build on SpectralNet:**
- **SCAN (2020)** — "Learning to Classify Images without Labels." Takes a different approach to deep clustering — learns neighbors first, then uses them for classification — but addresses the same scalability problem.
- **Deep Spectral Methods (2022)** — Uses DINO features + spectral clustering without training a separate network. Shows that modern self-supervised features can substitute for learned affinity.
- **SPICE (2021)** — "Semantic Pseudo-labeling for Image Clustering." Combines spectral-style embedding with prototype learning, achieving strong results without explicit eigenvector computation.

---

*Implementation available at [spectral_net/](https://github.com/ursaj123/PapersCode/tree/main/src/paperscode/spectral_net). Paper: Shaham et al., "[SpectralNet: Spectral Clustering Using Deep Neural Networks," ICLR 2018](https://arxiv.org/pdf/1801.01587).*