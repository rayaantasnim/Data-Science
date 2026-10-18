import os
import sys
import math
import random
import logging
import traceback
from typing import List, Tuple, Optional, Dict, Any, Union
import numpy as np
import scipy.spatial.distance as distance

# Configure production logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("StreamingiForestPipeline")

class StreamingWindow:
    """
    High-performance, zero-copy rolling window manager natively backed by NumPy.
    Efficiently manages continuous multidimensional data streams using a circular buffer design.
    """
    def __init__(self, capacity: int, dimensions: int) -> None:
        try:
            self.capacity = int(capacity)
            self.dimensions = int(dimensions)
            self.buffer = np.zeros((self.capacity, self.dimensions), dtype=np.float64)
            self.head = 0
            self.size = 0
            logger.info(f"Initialized StreamingWindow with capacity={capacity}, dimensions={dimensions}")
        except Exception as e:
            logger.error(f"Failed to initialize StreamingWindow: {e}")
            traceback.print_exc()
            raise

    def append(self, point: np.ndarray) -> None:
        """Appends a new multidimensional data point to the rolling window buffer."""
        try:
            pt = np.asarray(point, dtype=np.float64)
            if pt.shape != (self.dimensions,):
                raise ValueError(f"Expected shape ({self.dimensions},), got {pt.shape}")
            
            self.buffer[self.head] = pt
            self.head = (self.head + 1) % self.capacity
            if self.size < self.capacity:
                self.size += 1
        except Exception as e:
            logger.error(f"Error appending point to StreamingWindow: {e}")
            traceback.print_exc()
            raise

    def get_window_data(self) -> np.ndarray:
        """Returns the currently active rolling window data in chronological order (zero-copy view where possible)."""
        try:
            if self.size < self.capacity:
                return self.buffer[:self.size]
            else:
                return np.concatenate((self.buffer[self.head:], self.buffer[:self.head]), axis=0)
        except Exception as e:
            logger.error(f"Error retrieving window data: {e}")
            traceback.print_exc()
            return np.empty((0, self.dimensions), dtype=np.float64)

class IsolationTreeNode:
    """Node structure for the Online Isolation Tree supporting recursive partitions and streaming metadata."""
    def __init__(self, height: int, max_height: int) -> None:
        self.height = height
        self.max_height = max_height
        self.feature: Optional[int] = None
        self.value: Optional[float] = None
        self.left: Optional['IsolationTreeNode'] = None
        self.right: Optional['IsolationTreeNode'] = None
        self.size: int = 0
        self.is_leaf: bool = True

    def fit(self, X: np.ndarray) -> None:
        """Recursively builds or updates the isolation tree node partitions."""
        try:
            n_samples, n_features = X.shape
            self.size = n_samples

            if self.height >= self.max_height or n_samples <= 1 or np.all(X == X[0]):
                self.is_leaf = True
                return

            self.is_leaf = False
            self.feature = random.randint(0, n_features - 1)
            feat_min = float(np.min(X[:, self.feature]))
            feat_max = float(np.max(X[:, self.feature]))

            if feat_min == feat_max:
                self.is_leaf = True
                return

            self.value = random.uniform(feat_min, feat_max)
            
            left_mask = X[:, self.feature] < self.value
            right_mask = ~left_mask

            # Handle edge case where split isolates all points to one side
            if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                self.is_leaf = True
                return

            self.left = IsolationTreeNode(self.height + 1, self.max_height)
            self.left.fit(X[left_mask])

            self.right = IsolationTreeNode(self.height + 1, self.max_height)
            self.right.fit(X[right_mask])

        except Exception as e:
            logger.error(f"Error during node fitting: {e}")
            traceback.print_exc()
            self.is_leaf = True

    def path_length(self, point: np.ndarray, current_height: float = 0.0) -> float:
        """Computes the traversal path length for a given sample vector."""
        try:
            if self.is_leaf or self.left is None or self.right is None:
                # Average path length adjustment for un-isolated points exceeding max height
                return current_height + self._c_factor(self.size)

            if point[self.feature] < self.value:
                return self.left.path_length(point, current_height + 1.0)
            else:
                return self.right.path_length(point, current_height + 1.0)
        except Exception as e:
            logger.error(f"Error calculating path length: {e}")
            traceback.print_exc()
            return current_height

    @staticmethod
    def _c_factor(n: int) -> float:
        """Calculates average path length of unsuccessful searches in a Binary Search Tree (BST)."""
        try:
            if n <= 1:
                return 0.0
            if n == 2:
                return 1.0
            return 2.0 * (math.log(n - 1) + 0.5772156649) - (2.0 * (n - 1) / n)
        except Exception as e:
            logger.error(f"Error calculating c_factor for n={n}: {e}")
            return 0.0

class OnlineIsolationTree:
    """Wrapper for a single Online Isolation Tree instance."""
    def __init__(self, max_samples: int) -> None:
        self.max_samples = max_samples
        self.max_height = int(math.ceil(math.log2(max(self.max_samples, 2))))
        self.root: Optional[IsolationTreeNode] = None

    def fit(self, X: np.ndarray) -> None:
        """Fits the tree on a subsampled subset of the rolling window matrix."""
        try:
            n_samples = X.shape[0]
            indices = np.random.choice(n_samples, min(self.max_samples, n_samples), replace=False)
            subsample = X[indices]
            
            self.root = IsolationTreeNode(0, self.max_height)
            self.root.fit(subsample)
        except Exception as e:
            logger.error(f"Error fitting OnlineIsolationTree: {e}")
            traceback.print_exc()

    def path_length(self, point: np.ndarray) -> float:
        """Returns path length for an individual sample vector."""
        try:
            if self.root is None:
                return 0.0
            return self.root.path_length(point)
        except Exception as e:
            logger.error(f"Error in tree path_length evaluation: {e}")
            traceback.print_exc()
            return 0.0

class ConceptDriftDetector:
    """Monitors rolling statistical distributions to detect out-of-distribution concept drift."""
    def __init__(self, window_size: int, threshold_multiplier: float = 2.5) -> None:
        self.window_size = window_size
        self.threshold_multiplier = threshold_multiplier
        self.baseline_mean: Optional[np.ndarray] = None
        self.baseline_cov: Optional[np.ndarray] = None
        self.initialized = False

    def update(self, X: np.ndarray) -> bool:
        """
        Evaluates current data chunk for drift against baseline statistics using Mahalanobis distance metrics.
        Returns True if drift is detected.
        """
        try:
            if len(X) < 10:
                return False

            current_mean = np.mean(X, axis=0)
            current_cov = np.cov(X, rowvar=False)

            if not self.initialized:
                self.baseline_mean = current_mean
                self.baseline_cov = current_cov
                # Regularize covariance matrix to ensure invertibility
                self.baseline_cov += np.eye(self.baseline_cov.shape[0]) * 1e-5
                self.initialized = True
                return False

            # Compute Mahalanobis distance between current mean and baseline mean
            inv_cov = np.linalg.pinv(self.baseline_cov)
            diff = current_mean - self.baseline_mean
            m_dist = np.sqrt(max(0.0, np.dot(np.dot(diff.T, inv_cov), diff)))

            # Dynamic thresholding based on feature dimensionality scaling
            threshold = self.threshold_multiplier * np.sqrt(X.shape[1])

            if m_dist > threshold:
                logger.warning(f"Concept Drift Detected! Mahalanobis distance: {m_dist:.4f} > threshold: {threshold:.4f}")
                # Reset baseline to adapt to new state
                self.baseline_mean = current_mean
                self.baseline_cov = current_cov + np.eye(current_cov.shape[0]) * 1e-5
                return True

            return False
        except Exception as e:
            logger.error(f"Error in ConceptDriftDetector update: {e}")
            traceback.print_exc()
            return False

class OnlineIsolationForest:
    """
    Production-grade Online Isolation Forest architecture built completely from scratch
    incorporating real-time anomaly scoring, rolling window management, and concept drift adaptation.
    """
    def __init__(self, n_estimators: int = 100, max_samples: int = 256, window_capacity: int = 2000, dimensions: int = 5) -> None:
        try:
            self.n_estimators = int(n_estimators)
            self.max_samples = int(max_samples)
            self.window = StreamingWindow(window_capacity, dimensions)
            self.trees: List[OnlineIsolationTree] = []
            self.drift_detector = ConceptDriftDetector(window_capacity)
            self.dimensions = dimensions
            self.is_trained = False
            logger.info(f"Initialized OnlineIsolationForest with n_estimators={n_estimators}, max_samples={max_samples}")
        except Exception as e:
            logger.error(f"Failed to initialize OnlineIsolationForest: {e}")
            traceback.print_exc()
            raise

    def _initialize_forest(self, X: np.ndarray) -> None:
        """Initializes and trains ensemble trees on current window data."""
        try:
            logger.info("Initializing forest ensemble trees...")
            self.trees = [OnlineIsolationTree(self.max_samples) for _ in range(self.n_estimators)]
            for tree in self.trees:
                tree.fit(X)
            self.is_trained = True
            logger.info("Forest ensemble successfully trained.")
        except Exception as e:
            logger.error(f"Error initializing forest ensemble: {e}")
            traceback.print_exc()

    def process_point(self, point: np.ndarray) -> Tuple[float, bool]:
        """
        Ingests a streaming point, updates the window, checks for concept drift,
        re-indexes the forest if drift occurs, and computes the real-time anomaly score.
        """
        try:
            pt = np.asarray(point, dtype=np.float64)
            self.window.append(pt)
            window_data = self.window.get_window_data()

            drift_triggered = False
            if len(window_data) >= self.max_samples:
                # Check for concept drift periodically
                if self.drift_detector.update(window_data):
                    logger.info("Triggering seamless partial/full forest re-indexing due to concept drift...")
                    self._initialize_forest(window_data)
                    drift_triggered = True

            if not self.is_trained:
                if len(window_data) >= self.max_samples:
                    self._initialize_forest(window_data)
                return 0.5, drift_triggered  # Default neutral score during warm-up phase

            # Compute anomaly score across ensemble trees
            paths = [tree.path_length(pt) for tree in self.trees]
            mean_path_length = np.mean(paths)
            
            # Normalization factor c(n)
            c = IsolationTreeNode._c_factor(min(len(window_data), self.max_samples))
            
            if c == 0:
                score = 0.5
            else:
                score = 2.0 ** (-mean_path_length / c)

            return float(score), drift_triggered
        except Exception as e:
            logger.error(f"Error processing streaming point: {e}")
            traceback.print_exc()
            return 0.0, False

if __name__ == "__main__":
    try:
        logger.info("Starting Streaming Isolation Forest Pipeline Execution Simulation...")

        dimensions = 4
        # Initialize pipeline with 50 trees, max_samples=128, window capacity=1000
        pipeline = OnlineIsolationForest(
            n_estimators=50,
            max_samples=128,
            window_capacity=1000,
            dimensions=dimensions
        )

        logger.info("Phase 1: Streaming Normal Data Stream (Warm-up & Baseline)...")
        # Generate baseline normal distribution data points
        np.random.seed(42)
        normal_data = np.random.normal(loc=0.0, scale=1.0, size=(300, dimensions))

        scores = []
        for i, point in enumerate(normal_data):
            score, drift = pipeline.process_point(point)
            scores.append(score)
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1} normal points. Last anomaly score: {score:.4f}")

        logger.info("Phase 2: Injecting Adversarial / Out-of-Distribution Anomalies...")
        # Inject extreme outliers / anomalies
        anomaly_points = np.random.normal(loc=8.0, scale=0.5, size=(20, dimensions))

        anomaly_scores = []
        for i, point in enumerate(anomaly_points):
            score, drift = pipeline.process_point(point)
            anomaly_scores.append(score)
            logger.info(f"Anomaly Point {i+1} Injected | Score: {score:.4f} (Spike detected: {score > 0.6})")

        logger.info("Phase 3: Demonstrating Concept Drift & Automated Adaptation Phase...")
        # Shift data distribution significantly to trigger concept drift detector
        drift_data = np.random.normal(loc=5.0, scale=1.2, size=(250, dimensions))

        drift_scores = []
        for i, point in enumerate(drift_data):
            score, drift = pipeline.process_point(point)
            drift_scores.append(score)
            if drift:
                logger.info(f"Drift adaptation event handled successfully at step {i+1}. Current score: {score:.4f}")

        logger.info("Pipeline execution simulation completed successfully without errors.")

    except Exception as e:
        logger.critical(f"Critical unhandled exception in main pipeline execution: {e}")
        traceback.print_exc()
        sys.exit(1)