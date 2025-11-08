---
name: evolutionary-clustering-guide
description: Teaches evolutionary clustering from fundamentals (K-means, DBSCAN) to advanced (Mixture of Typicalities). Iterative learning with practical examples for IoT IDS context.
version: 1.0.0
author: Research Acceleration System
activate_when:
  - "clustering"
  - "evolutionary clustering"
  - "Maia et al"
  - "Mixture of Typicalities"
  - "concept drift"
  - "Phase 2"
---

# Evolutionary Clustering Guide

## Purpose

Teach evolutionary clustering through iterative learning cycles: basic concept → implement simple version → experiment → understand deeper → implement advanced version.

**Target:** Augusto needs to learn clustering from basics (K-means, DBSCAN) up to Mixture of Typicalities for Phase 2 of research.

## Learning Path

### Level 1: Clustering Fundamentals (Week 1)

**Concepts to Learn:**
- What is clustering? (grouping similar data points)
- Distance metrics (Euclidean, Manhattan, Cosine)
- K-means algorithm: centroids, assignment, update loop
- DBSCAN: density-based, epsilon neighborhood, core points
- Evaluation metrics: Silhouette Score, Davies-Bouldin Index

**Practical Approach:**
1. **Tiny Example First:** Cluster 100 samples from CICIoT2023 using sklearn
2. **Visualize:** Plot clusters in 2D (PCA reduction)
3. **Experiment:** Try different K values, see what happens
4. **Connect to IDS:** "Normal traffic" vs "attack types" as clusters

**Code Template:**
```python
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

# Load tiny sample
X_train = np.load('data/processed/binary/X_train_binary.npy')[:100]

# K-means
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_train)

# Visualize in 2D
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_train)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='viridis')
plt.title('K-means Clustering Visualization')
plt.show()
```

**Exercises:**
- Vary K from 2 to 10, plot Silhouette scores
- Compare K-means vs DBSCAN on same data
- Which algorithm finds "normal" vs "attack" better?

**Time:** ~6-8 hours spread over 2-3 days

---

### Level 2: Temporal Clustering (Week 2)

**Concepts to Learn:**
- Why static clustering fails with concept drift
- Temporal windows: dividing data by time
- Cluster evolution: how clusters change over time
- Cluster tracking: matching old clusters to new clusters

**Practical Approach:**
1. **Simulate Drift:** Create temporal windows from CICIoT2023
2. **Track Changes:** Run K-means on each window, compare centroids
3. **Visualize Evolution:** Plot how clusters move over time
4. **Problem Discovery:** See where static approach breaks

**Code Template:**
```python
import pandas as pd

# Assuming CICIoT2023 has timestamp column
df = pd.read_csv('data/processed/sampled.csv')
df = df.sort_values('timestamp')

# Split into temporal windows (e.g., hourly)
windows = [df[i:i+1000] for i in range(0, len(df), 1000)]

# Run K-means on each window
centroids_over_time = []
for window in windows:
    X_window = window.drop(['Label', 'timestamp'], axis=1).values
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X_window)
    centroids_over_time.append(kmeans.cluster_centers_)

# Calculate centroid drift (distance between consecutive windows)
drifts = []
for i in range(1, len(centroids_over_time)):
    drift = np.linalg.norm(centroids_over_time[i] - centroids_over_time[i-1])
    drifts.append(drift)

plt.plot(drifts)
plt.title('Concept Drift Magnitude Over Time')
plt.xlabel('Time Window')
plt.ylabel('Centroid Drift')
plt.show()
```

**Experiments:**
- Identify periods of high concept drift
- Correlate drift with attack types in dataset
- See if drift detection can signal attacks

**Time:** ~8-10 hours spread over 3-4 days

---

### Level 3: Evolutionary Clustering Concepts (Week 3)

**Concepts to Learn:**
- What makes clustering "evolutionary"
- Temporal smoothness: penalizing rapid changes
- Cost function with history: current fit + temporal consistency
- Online learning: updating clusters incrementally

**Maia et al. (2020) Core Ideas:**
- Mixture of Typicalities: clusters represent "typical" patterns
- Typicality degree: how well a point fits a cluster (fuzzy membership)
- Evolution mechanism: clusters adapt gradually, not abruptly
- Forgetting factor: recent data has more weight

**Simplified Explanation:**
```
Traditional K-means: "Forget everything, recluster from scratch"
Evolutionary: "Remember previous clusters, adapt gradually"

Analogy:
- K-means = amnesia patient, starts fresh every time
- Evolutionary = person who learns from history
```

**Mathematical Intuition (no need to memorize):**
```
Typicality T(x, cluster_i) = function of distance to centroid
Lower distance → higher typicality (closer to "typical" pattern)

Cluster update:
new_centroid = (1-α) * old_centroid + α * new_data_mean
where α = learning rate (controls adaptation speed)
```

**Practical Approach:**
1. **Read Maia et al. 2020 paper** with `paper-reading-accelerator` skill
2. **Extract pseudocode** from paper (skip heavy math initially)
3. **Design simple version:** K-means + exponential moving average of centroids
4. **Plan implementation:** What data structures needed? What functions?

**Deliverable:** Design document in `docs/plans/2025-XX-XX-evolutionary-clustering-design.md`

**Time:** ~10-12 hours (reading + designing)

---

### Level 4: Implementing Mixture of Typicalities (Weeks 4-6)

**Goal:** Build working implementation of Maia et al. algorithm

**Iterative Implementation:**

**Sprint 1 (Week 4): Basic Structure**
```python
# src/clustering/evolutionary_clustering.py

class EvolutionaryClusterer:
    def __init__(self, n_clusters=3, alpha=0.3, forgetting_factor=0.95):
        """
        Args:
            n_clusters: Number of clusters
            alpha: Learning rate for centroid updates
            forgetting_factor: Weight of historical data
        """
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.forgetting_factor = forgetting_factor
        self.centroids = None
        self.history = []

    def fit_initial(self, X):
        """Initialize with standard K-means"""
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_clusters)
        kmeans.fit(X)
        self.centroids = kmeans.cluster_centers_
        return self

    def update(self, X_new):
        """Update clusters with new data batch"""
        # 1. Calculate typicalities for each point
        typicalities = self._calculate_typicalities(X_new)

        # 2. Update centroids using exponential moving average
        new_centroids = self._compute_new_centroids(X_new, typicalities)
        self.centroids = (1 - self.alpha) * self.centroids + self.alpha * new_centroids

        # 3. Store history
        self.history.append(self.centroids.copy())

        return self

    def _calculate_typicalities(self, X):
        """Calculate typicality degree for each point to each cluster"""
        # Simplified: inverse distance
        distances = np.array([[np.linalg.norm(x - c) for c in self.centroids] for x in X])
        typicalities = 1 / (1 + distances)  # Higher typicality = closer
        return typicalities

    def _compute_new_centroids(self, X, typicalities):
        """Weighted average of points based on typicalities"""
        new_centroids = []
        for i in range(self.n_clusters):
            weights = typicalities[:, i]
            weighted_sum = np.sum(X * weights[:, np.newaxis], axis=0)
            new_centroid = weighted_sum / np.sum(weights)
            new_centroids.append(new_centroid)
        return np.array(new_centroids)

    def predict(self, X):
        """Assign points to closest cluster"""
        distances = np.array([[np.linalg.norm(x - c) for c in self.centroids] for x in X])
        return np.argmin(distances, axis=1)
```

**Test it:**
```python
# Test with temporal windows
clusterer = EvolutionaryClusterer(n_clusters=3, alpha=0.3)
clusterer.fit_initial(windows[0])

for window in windows[1:]:
    clusterer.update(window)

# Compare with retraining K-means each time
```

**Sprint 2 (Week 5): Add Full Maia et al. Features**
- Proper typicality function (Gaussian-based, not just inverse distance)
- Forgetting factor implementation
- Cluster merging/splitting when needed
- Validation on CICIoT2023

**Sprint 3 (Week 6): Integration & Experiments**
- Integrate with DVC pipeline
- Run experiments: evolutionary vs static K-means vs DBSCAN
- Metrics: Silhouette score over time, adaptation speed, detection accuracy
- Generate plots and analysis

---

### Level 5: Validation & Analysis (Weeks 7-9)

**Experiments to Run:**

1. **Concept Drift Adaptation:**
   - Inject concept drift (mix different attack types over time)
   - Measure: How quickly does clustering adapt?
   - Compare: Evolutionary vs retraining K-means

2. **Detection Performance:**
   - Use cluster labels for anomaly detection
   - Metrics: Precision, Recall, F1 (compare with Phase 1 baseline)
   - Question: Does evolutionary approach improve over static?

3. **Computational Efficiency:**
   - Time per update vs full retraining
   - Memory usage
   - Scalability test (vary data size)

4. **Sensitivity Analysis:**
   - Vary α (learning rate): 0.1, 0.3, 0.5, 0.7
   - Vary forgetting_factor: 0.9, 0.95, 0.99
   - Find optimal parameters for CICIoT2023

**Deliverable:** Results section for dissertation chapter

---

## Teaching Approach

### When Augusto Asks "How does X work?"

**Bad response:** Long theoretical explanation with equations
**Good response:**
1. One-sentence intuition
2. Tiny code example (5-10 lines)
3. "Try it yourself" task
4. "What did you observe?" discussion

### Example Dialogue:

**Augusto:** "What is typicality in clustering?"

**You:**
"Typicality = how 'typical' a point is for a cluster. High typicality = very typical, low = outlier.

Quick code:
```python
distance = np.linalg.norm(point - centroid)
typicality = 1 / (1 + distance)  # Close → high typicality
```

Try this: Calculate typicalities for 5 points to 2 cluster centers. Which points are 'typical' for which cluster?"

[Wait for Augusto to try]

---

## Connection to IoT IDS

**Always relate to research goal:**

- **Normal traffic** = one cluster (tight, consistent)
- **Different attack types** = different clusters
- **New/evolving attacks** = concept drift (clusters shift)
- **Evolutionary clustering** = adapts to new attack patterns without retraining

**Phase 2 Success Criteria:**
- Evolutionary clustering detects attacks as well as Phase 1 baseline (F1 > 0.99)
- Adapts to concept drift faster than retraining approach
- Lower computational cost than full retraining
- Ready to integrate with streaming (Phase 3)

---

## Resources

**Papers (in Zotero):**
- Maia et al. (2020) - Core algorithm
- Lu et al. (2019) - Concept drift theory
- Wahab (2022) - IoT concept drift detection

**Code Examples:**
- scikit-learn clustering documentation
- River library (online machine learning)

**Validation:**
- Use Phase 1 experiment framework (5 runs, grid search, MLflow tracking)
- Compare with Phase 1 baseline algorithms

---

## Checkpoints

After each level, validate understanding:

**Level 1:** Can Augusto run K-means and DBSCAN on CICIoT2023? ✅
**Level 2:** Can he visualize concept drift in temporal windows? ✅
**Level 3:** Does he understand evolutionary clustering conceptually? ✅
**Level 4:** Is basic implementation working? ✅
**Level 5:** Are experiments complete with good results? ✅

---

**Use this skill to guide Phase 2 implementation with hands-on, iterative learning.**
