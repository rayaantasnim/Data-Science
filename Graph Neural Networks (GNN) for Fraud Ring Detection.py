import os
import sys
import math
import logging
import traceback
from typing import Tuple, Dict, Any, Optional

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("FraudGCNArchitect")


class SyntheticFraudGraphGenerator:
    """
    Generates synthetic transaction graphs representing financial entities,
    fraud rings, and legitimate connection topologies using NumPy and SciPy.
    """
    def __init__(self, num_nodes: int = 1000, num_features: int = 32, fraud_ratio: float = 0.1, seed: int = 42):
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.fraud_ratio = fraud_ratio
        self.seed = seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

    def generate_graph(self) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray]:
        """
        Builds feature matrix X, adjacency matrix A, and binary labels y.
        Encapsulates error handling and structural validation.
        """
        try:
            logger.info("Initializing synthetic transaction graph generation...")
            
            # 1. Feature Matrix (X): Normal distribution with some structured noise for fraud patterns
            num_fraud = int(self.num_nodes * self.fraud_ratio)
            num_legit = self.num_nodes - num_fraud
            
            # Legitimate nodes feature distribution
            X_legit = np.random.normal(loc=0.0, scale=1.0, size=(num_legit, self.num_features))
            # Fraudulent nodes feature distribution (shifted mean to simulate behavioral anomalies)
            X_fraud = np.random.normal(loc=2.5, scale=1.5, size=(num_fraud, self.num_features))
            
            X = np.vstack([X_legit, X_fraud])
            
            # Labels: 0 for legitimate, 1 for fraudulent
            y = np.zeros(self.num_nodes, dtype=np.int64)
            y[num_legit:] = 1
            
            # Shuffle nodes to blend fraud rings
            indices = np.arange(self.num_nodes)
            np.random.shuffle(indices)
            X = X[indices]
            y = y[indices]
            
            # Inverse map to track original indices if needed, but not strictly necessary for unsupervised topology
            # 2. Adjacency Matrix (A): Construct scale-free / community-based connections
            # Using block model to simulate tightly-knit fraud rings and sparse cross-links
            adj_dense = np.zeros((self.num_nodes, self.num_nodes), dtype=np.float32)
            
            # Create random Erdős–Rényi background connections
            p_background = 0.005
            background_edges = np.random.rand(self.num_nodes, self.num_nodes) < p_background
            adj_dense[background_edges] = 1.0
            
            # Create dense fraud rings (interconnected cliques or clusters)
            fraud_indices = np.where(y == 1)[0]
            if len(fraud_indices) > 1:
                for i in range(len(fraud_indices)):
                    for j in range(i + 1, len(fraud_indices)):
                        # High connectivity probability within fraud rings
                        if np.random.rand() < 0.4:
                            idx1 = fraud_indices[i]
                            idx2 = fraud_indices[j]
                            adj_dense[idx1, idx2] = 1.0
                            adj_dense[idx2, idx1] = 1.0
                            
            # Ensure symmetry and remove self loops initially
            adj_dense = np.maximum(adj_dense, adj_dense.T)
            np.fill_diagonal(adj_dense, 0.0)
            
            # Convert to SciPy CSR sparse matrix
            A = sp.csr_matrix(adj_dense)
            logger.info(f"Successfully generated graph with {self.num_nodes} nodes, {A.nnz} edges, and {num_fraud} fraud nodes.")
            return A, X, y

        except Exception as e:
            logger.error(f"Error during graph generation: {str(e)}")
            logger.error(traceback.format_exc())
            raise


class GraphConvolution(nn.Module):
    """
    Custom Graph Convolutional Layer implemented from scratch using PyTorch primitives.
    Calculates: Z = sigma(D_hat^{-1/2} * A_hat * D_hat^{-1/2} * X * W)
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(GraphConvolution, self).__init__()
        try:
            self.in_features = in_features
            self.out_features = out_features
            self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
            if bias:
                self.bias = nn.Parameter(torch.FloatTensor(out_features))
            else:
                self.register_parameter('bias', None)
            self.reset_parameters()
        except Exception as e:
            logger.error(f"Initialization error in GraphConvolution: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def reset_parameters(self):
        """Custom Xavier/Glorot initialization coefficients tailored for graph architectures."""
        try:
            stdv = 1.0 / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.data.uniform_(-stdv, stdv)
        except Exception as e:
            logger.error(f"Error resetting parameters: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def forward(self, x: torch.Tensor, adj_normalized: torch.Tensor) -> torch.Tensor:
        """
        Forward pass executing sparse matrix multiplication and message passing.
        """
        try:
            # Linear transformation: X * W
            support = torch.mm(x, self.weight)
            
            # Message passing via normalized adjacency tensor: D^{-1/2} * A * D^{-1/2} * (X * W)
            # Utilizing torch.sparse.mm for sparse-dense efficiency
            output = torch.sparse.mm(adj_normalized, support)
            
            if self.bias is not None:
                output = output + self.bias
            return output
        except Exception as e:
            logger.error(f"Error during GraphConvolution forward pass: {str(e)}")
            logger.error(traceback.format_exc())
            raise


class GCNFraudDetector(nn.Module):
    """
    2-Layer Graph Convolutional Network for Semi-Supervised Node Classification
    specialized in identifying interconnected fraud rings.
    """
    def __init__(self, nfeat: int, nhid: int, nclass: int, dropout: float = 0.5):
        super(GCNFraudDetector, self).__init__()
        try:
            self.gc1 = GraphConvolution(nfeat, nhid)
            self.gc2 = GraphConvolution(nhid, nclass)
            self.dropout = dropout
        except Exception as e:
            logger.error(f"Error initializing GCNFraudDetector architecture: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def forward(self, x: torch.Tensor, adj_normalized: torch.Tensor) -> torch.Tensor:
        try:
            x = F.relu(self.gc1(x, adj_normalized))
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.gc2(x, adj_normalized)
            return F.log_softmax(x, dim=1)
        except Exception as e:
            logger.error(f"Error in GCNFraudDetector forward propagation: {str(e)}")
            logger.error(traceback.format_exc())
            raise


class GraphPreprocessor:
    """
    Handles graph normalization mathematics from spectral graph theory:
    Â = A + I (adding self-loops) and symmetric normalization D^{-1/2} Â D^{-1/2}.
    """
    @staticmethod
    def normalize_adjacency(adj: sp.csr_matrix) -> torch.sparse.FloatTensor:
        try:
            logger.info("Computing symmetric normalization for adjacency matrix...")
            # Add self-loops: Â = A + I
            adj_with_self_loops = adj + sp.eye(adj.shape[0])
            
            # Calculate degree matrix D
            row_sum = np.array(adj_with_self_loops.sum(1)).flatten()
            
            # D^{-1/2}
            with np.errstate(divide='ignore'):
                row_sum_inv_sqrt = np.power(row_sum, -0.5)
            row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.0
            
            deg_mat_inv_sqrt = sp.diags(row_sum_inv_sqrt)
            
            # Symmetric normalization: D^{-1/2} * Â * D^{-1/2}
            normalized_adj = deg_mat_inv_sqrt.dot(adj_with_self_loops).dot(deg_mat_inv_sqrt)
            
            # Convert SciPy sparse matrix to PyTorch sparse tensor
            normalized_adj = normalized_adj.tocoo()
            indices = torch.LongTensor(np.vstack((normalized_adj.row, normalized_adj.col)))
            values = torch.FloatTensor(normalized_adj.data)
            shape = torch.Size(normalized_adj.shape)
            
            sparse_tensor = torch.sparse_coo_tensor(indices, values, shape).coalesce()
            logger.info("Adjacency normalization complete and converted to PyTorch sparse tensor.")
            return sparse_tensor
        except Exception as e:
            logger.error(f"Error during graph normalization: {str(e)}")
            logger.error(traceback.format_exc())
            raise


def evaluate_cluster_metrics(output: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    """
    Evaluates binary node classifications (Fraudulent vs. Legitimate) across nodes using final layer embeddings.
    """
    try:
        preds = output[mask].max(1)[1]
        target = labels[mask]
        
        preds_np = preds.cpu().numpy()
        target_np = target.cpu().numpy()
        
        correct = (preds_np == target_np).sum()
        acc = correct / len(target_np) if len(target_np) > 0 else 0.0
        
        # Calculate Precision, Recall, F1 for fraud class (class 1)
        true_positives = ((preds_np == 1) & (target_np == 1)).sum()
        false_positives = ((preds_np == 1) & (target_np == 0)).sum()
        false_negatives = ((preds_np == 0) & (target_np == 1)).sum()
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "accuracy": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }
    except Exception as e:
        logger.error(f"Error evaluating cluster metrics: {str(e)}")
        logger.error(traceback.format_exc())
        raise


def main():
    """
    Runnable execution entry point building a sparse structural transaction graph,
    initializing the GCN, running optimization loops, and demonstrating fraud ring isolation.
    """
    try:
        logger.info("=== Starting GCN Fraud Ring Detection Pipeline ===")
        
        # Hyperparameters
        num_nodes = 1200
        num_features = 64
        hidden_dim = 32
        num_classes = 2
        epochs = 100
        lr = 0.01
        weight_decay = 5e-4
        
        # 1. Generate Synthetic Fraud Graph
        generator = SyntheticFraudGraphGenerator(num_nodes=num_nodes, num_features=num_features, fraud_ratio=0.15)
        adj, features, labels = generator.generate_graph()
        
        # 2. Normalize Adjacency and Prepare Tensors
        adj_normalized = GraphPreprocessor.normalize_adjacency(adj)
        
        features_tensor = torch.FloatTensor(features)
        labels_tensor = torch.LongTensor(labels)
        
        # Create Semi-Supervised Masks (Train, Validation, Test splits)
        indices = np.random.permutation(num_nodes)
        train_size = int(num_nodes * 0.6)
        val_size = int(num_nodes * 0.2)
        
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        train_mask[indices[:train_size]] = True
        val_mask[indices[train_size:train_size + val_size]] = True
        test_mask[indices[train_size + val_size:]] = True
        
        # 3. Initialize GCN Model & Optimizer
        model = GCNFraudDetector(nfeat=num_features, nhid=hidden_dim, nclass=num_classes, dropout=0.3)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        logger.info("Starting model training optimization loop...")
        model.train()
        
        for epoch in range(epochs):
            try:
                optimizer.zero_grad()
                output = model(features_tensor, adj_normalized)
                
                # Custom optimization objective: Masked Cross-Entropy Loss
                loss = F.nll_loss(output[train_mask], labels_tensor[train_mask])
                
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    model.eval()
                    with torch.no_grad():
                        val_output = model(features_tensor, adj_normalized)
                        val_loss = F.nll_loss(val_output[val_mask], labels_tensor[val_mask]).item()
                        val_metrics = evaluate_cluster_metrics(val_output, labels_tensor, val_mask)
                    model.train()
                    logger.info(
                        f"Epoch {epoch + 1}/{epochs} | Train Loss: {loss.item():.4f} | "
                        f"Val Loss: {val_loss:.4f} | Val F1-Score: {val_metrics['f1_score']:.4f}"
                    )
            except Exception as inner_e:
                logger.error(f"Error during epoch {epoch + 1}: {str(inner_e)}")
                raise inner_e
                
        # 4. Final Evaluation on Unseen Test Nodes
        logger.info("Executing final cluster auditing on test partition...")
        model.eval()
        with torch.no_grad():
            final_output = model(features_tensor, adj_normalized)
            test_metrics = evaluate_cluster_metrics(final_output, labels_tensor, test_mask)
            
        logger.info("=== Final Evaluation Results (Fraud Ring Isolation) ===")
        logger.info(f"Test Accuracy:  {test_metrics['accuracy'] * 100:.2f}%")
        logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
        logger.info(f"Test Recall:    {test_metrics['recall']:.4f}")
        logger.info(f"Test F1-Score:  {test_metrics['f1_score']:.4f}")
        logger.info("=== Pipeline Execution Complete Successfully ===")

    except Exception as e:
        logger.error(f"Critical execution failure in main pipeline: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()