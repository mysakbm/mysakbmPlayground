# Lecture: Clustering Techniques in Machine Learning

## 1. Introduction

**Definition:**  
Clustering is the process of grouping a set of unlabeled data points into subsets (clusters) where points in the same cluster are more similar to each other than to those in other clusters.

**Mathematical Setup:**  
Given $X = \{x_1, x_2, \dots, x_n\}$ with each $x_i \in \mathbb{R}^d$, clustering aims to partition $X$ into $k$ disjoint subsets (clusters) $C = \{C_1, C_2, \dots, C_k\}$ such that:  
- $C_i \cap C_j = \emptyset$ for all $i \neq j$  
- $\bigcup_{i=1}^k C_i = X$

**Applications:**  
- Customer segmentation  
- Image compression and object grouping  
- Document/topic clustering in NLP  
- Bioinformatics (gene expression data analysis)

---

## 2. Distance and Similarity Measures

A key component of clustering is defining how to measure similarity or distance between points.

**Common Metrics:**

1. **Euclidean Distance (L2):**  
   $
   d(x,y) = \sqrt{\sum_{m=1}^{d}(x_m - y_m)^2}
   $

2. **Manhattan Distance (L1):**  
   $
   d(x,y) = \sum_{m=1}^{d}|x_m - y_m|
   $

3. **Cosine Similarity:**  
   $
   \text{sim}(x,y) = \frac{x \cdot y}{\|x\|\|y\|}
   $

**Note:** Choice of metric can significantly influence clustering results, especially in high-dimensional spaces.

---

## 3. Major Clustering Families

| Method Type       | Examples               | Key Idea                                              |
|-------------------|------------------------|-------------------------------------------------------|
| Centroid-Based    | K-means, K-medoids     | Clusters represented by centers; minimize variance.   |
| Hierarchical      | Agglomerative, Divisive| Builds a tree-like structure (dendrogram).            |
| Density-Based     | DBSCAN, OPTICS         | Identifies dense regions as clusters, ignoring noise. |
| Distribution-Based| Gaussian Mixtures (GMM)| Assumes data from a mixture of probability distributions. |
| Graph-Based       | Spectral Clustering    | Uses graph Laplacians and eigenvectors to partition data. |
| Deep/Embedding-Based | DEC, VAE-based methods | Learns latent representations tailored for clustering. |

---

## 4. Centroid-Based Clustering

### 4.1 K-Means

**Objective:**  
Minimize the Within-Cluster Sum of Squares (WCSS):  
$
J = \sum_{j=1}^{k} \sum_{x_i \in C_j} \|x_i - \mu_j\|^2
$
where $\mu_j$ is the centroid of cluster $C_j$.

**Algorithm:**
1. Initialize $k$ centroids $\mu_j$.
2. Assign each point $x_i$ to the closest centroid:  
   $
   C_j^{(t)} = \{x_i : \|x_i - \mu_j^{(t)}\| \leq \|x_i - \mu_l^{(t)}\|, \forall l\}.
   $
3. Update centroids:  
   $
   \mu_j^{(t+1)} = \frac{1}{|C_j^{(t)}|}\sum_{x_i \in C_j^{(t)}} x_i.
   $
4. Repeat until convergence.

**Pros/Cons:**  
- Pros: Simple, fast, widely used.  
- Cons: Sensitive to initialization, primarily finds spherical clusters.

## 4.2 K-medoids

**K-medoids** is similar to K-means but uses actual data points as cluster centers (medoids) and can use arbitrary distance measures:

**Update Step:**  
Instead of the mean, choose the medoid $\tilde{\mu}_j$ as:
$
\tilde{\mu}_j = \arg\min_{x \in C_j} \sum_{x_i \in C_j} d(x_i, x)
$

This can be more robust to outliers compared to K-means.

---

## 5. Hierarchical Clustering

**Idea:**  
Build a hierarchy of clusters without specifying $k$ upfront.

**Agglomerative Clustering:**  
- Start with each point as its own cluster.  
- Iteratively merge the two "closest" clusters until only one cluster remains.

**Linkage Methods:**
- Single: $\displaystyle d(C_a, C_b) = \min_{x \in C_a, y \in C_b} d(x,y)$
- Complete: $\displaystyle d(C_a, C_b) = \max_{x \in C_a, y \in C_b} d(x,y)$
- Average: $\displaystyle d(C_a, C_b) = \frac{1}{|C_a||C_b|}\sum_{x \in C_a}\sum_{y \in C_b} d(x,y)$

**Pros/Cons:**  
- Pros: No need to pre-specify $k$, interpretable dendrogram.  
- Cons: High complexity, no backtracking once merged.

---

## 6. Density-Based Clustering: DBSCAN

**Concept:**  
Identifies "core" points in high-density regions and forms clusters around them.

**Parameters:**  
- **$\epsilon$**: Neighborhood radius  
- **MinPts**: Minimum number of points within $\epsilon$ for a point to be considered a "core" point.
- A point $p$ is a core point if at least $minPts$ points are within distance ε of it (including $p$).
- A point $q$ is directly reachable from $p$ if point $q$ is within distance $ε$ from core point $p$. Points are only said to be directly reachable from core points.
- A point $q$ is reachable from $p$ if there is a path $p_1, ..., p_n$ with $p_1 = p$ and $p_n = q$, where each $p_{i+1}$ is directly reachable from $p_i$. Note that this implies that the initial point and all points on the path must be core points, with the possible exception of $q$.
- All points not reachable from any other point are outliers or noise points.

**Algorithm Steps:**  
1. Find all core points.  
2. Form clusters by connecting all core points within $\epsilon$-distance.  
3. Assign non-core points that are within $\epsilon$ of a core point to that cluster.  
4. Points not reachable from any core point are labeled as noise.

**Pros & Cons:**  
- **Pros:** Finds arbitrarily shaped clusters, can detect noise/outliers.  
- **Cons:** Sensitive to parameters $\epsilon$ and MinPts, cannot handle varying density well.

**Density-Based Variants:**  
- **OPTICS:** Handles varying densities by producing an ordering of points.  
- **HDBSCAN:** Hierarchical approach to density-based clustering, no need to choose $\epsilon$.

**Pros/Cons:**  
- Pros: Detects arbitrarily shaped clusters, identifies outliers.  
- Cons: Requires suitable $\epsilon$ and MinPts, can struggle with varying densities.

---

## 7. Distribution-Based Clustering: Gaussian Mixture Models (GMM)

**Concept:**  
Assume data is generated from a mixture of $k$ Gaussian distributions:
$
p(x) = \sum_{j=1}^{k} \pi_j \mathcal{N}(x|\mu_j, \Sigma_j)
$
where $\pi_j$ are mixture weights, $\mu_j$ means, and $\Sigma_j$ covariances.

**Estimation Method: Expectation-Maximization (EM) Algorithm:**

1. **E-step:**
   Compute the posterior probability that point $x_i$ belongs to cluster $j$:
   $
   \gamma_{ij} = \frac{\pi_j \mathcal{N}(x_i|\mu_j,\Sigma_j)}{\sum_{l=1}^{k}\pi_l \mathcal{N}(x_i|\mu_l,\Sigma_l)}
   $

2. **M-step:**
   Update parameters:
   $
   \pi_j := \frac{1}{n}\sum_{i=1}^{n}\gamma_{ij}, \quad
   \mu_j := \frac{\sum_{i=1}^{n}\gamma_{ij}x_i}{\sum_{i=1}^{n}\gamma_{ij}}, \quad
   \Sigma_j := \frac{\sum_{i=1}^{n}\gamma_{ij}(x_i-\mu_j)(x_i-\mu_j)^T}{\sum_{i=1}^{n}\gamma_{ij}}
   $

**Pros & Cons:**  
- **Pros:** Can find more flexible cluster shapes (ellipsoidal), probabilistic interpretation.  
- **Cons:** May converge to local maxima, requires assumption of underlying Gaussianity.


## 8. Graph-Based Clustering: Spectral Clustering

**Idea:** Use eigenvectors of a similarity graph’s Laplacian matrix to cluster points.

**Steps:**
1. Construct a similarity graph $G$ from data points (e.g., using a Gaussian kernel):
   $
   w_{ij} = \exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right)
   $
2. Compute the graph Laplacian:
   $
   L = D - W, \quad D_{ii} = \sum_j w_{ij}
   $
3. Compute the eigenvectors of $L$ (or normalized versions of $L$).  
4. Use the top $k$ eigenvectors as features and apply K-means in this eigen-space.

**Pros & Cons:**  
- **Pros:** Can find clusters not linearly separable in the original space, very flexible.  
- **Cons:** Requires computation of eigenvectors (can be costly), sensitive to parameters in similarity computation.

---

## 9. Neural Network-Based Clustering

Modern methods integrate deep learning to produce suitable embeddings for clustering.

**Example: Deep Embedding Clustering (DEC)**  
- Jointly optimize a reconstruction loss (via autoencoder) and a clustering loss (e.g., Kullback–Leibler divergence).

**Loss Function:**
$
L = L_r + \lambda L_c
$
- Reconstruction loss: $\|X - \hat{X}\|_F^2$
- Clustering loss: A KL divergence comparing soft assignments $Q$ to a target distribution $P$.

---

## 10. Cluster Validation Metrics

**Internal Validation (no ground truth):**
- **Silhouette Coefficient:**  
  $
  s(i) = \frac{b(i)-a(i)}{\max\{a(i),b(i)\}}
  $
  where $a(i)$ = mean distance to points in same cluster,  
  $b(i)$ = min mean distance to points in another cluster.

- **Davies-Bouldin Index:**  
  $
  DB = \frac{1}{k}\sum_{i=1}^k \max_{j\neq i}\frac{\sigma_i + \sigma_j}{\| \mu_i - \mu_j \|}.
  $

**External Validation (with ground truth):**
- **Rand Index (RI):**  
  $
  RI = \frac{TP + TN}{TP + TN + FP + FN}.
  $

- **Adjusted Rand Index (ARI):** Adjusts RI for chance.

---

## 11. Complexity and Scalability

| Algorithm         | Complexity    | Notes                          |
|-------------------|---------------|--------------------------------|
| K-means           | $O(n k d)$ per iteration | Fast for large $n$, simple. |
| Hierarchical      | $O(n^3)$ (naive) | Complex, often for smaller datasets. |
| DBSCAN            | $O(n \log n)$ to $O(n^2)$ | Depends on indexing structure. |
| GMM (EM)          | $O(n k d^2)$ per iteration | Complexity depends on covariance structure. |
| Spectral Clustering| $O(n^3)$   | Dominated by eigen-decomposition. |

---

## 12. Advanced Techniques and Trends

- **Dimensionality Reduction (PCA, t-SNE, UMAP):**  
  Preprocessing to handle high-dimensional data.

- **Kernel Methods:**  
  Performing clustering in nonlinear feature spaces.

- **Fuzzy Clustering (e.g., Fuzzy C-means):**  
  Degrees of membership rather than a hard assignment:
  $
  J_m = \sum_{j=1}^k \sum_{i=1}^n u_{ij}^m \|x_i - \mu_j\|^2.
  $

- **Semi-Supervised Clustering:**  
  Incorporating constraints (must-link/cannot-link) or partial labels to guide clustering.

---

## 13. Conclusion

Clustering encompasses a wide range of algorithms and approaches. The choice of algorithm depends on:
- The shape and distribution of clusters
- The scale and dimensionality of the data
- The availability of domain knowledge or partial labels
- Required interpretability and complexity constraints 

Experimentation, dimensionality reduction, and appropriate validation are key to discovering meaningful structures.
