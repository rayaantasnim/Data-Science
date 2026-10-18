import os
import sys
import math
import logging
import traceback
from typing import Tuple, Optional, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# =====================================================================
# Logging Configuration
# =====================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("BayesianNeuralNetworkEngine")


# =====================================================================
# Custom Variational Linear Layer
# =====================================================================
class BayesianLinear(nn.Module):
    """
    Custom Variational Linear Layer for Bayesian Neural Networks.
    Treats each weight and bias parameter as a Gaussian distribution parameter pair:
    Mean (mu) and unconstrained Log Variance (rho), where sigma = softplus(rho).
    """
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0) -> None:
        super(BayesianLinear, self).__init__()
        try:
            self.in_features = in_features
            self.out_features = out_features
            self.prior_std = prior_std

            # Variational posterior parameters for weights
            self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
            self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))

            # Variational posterior parameters for biases
            self.bias_mu = nn.Parameter(torch.Tensor(out_features))
            self.bias_rho = nn.Parameter(torch.Tensor(out_features))

            self._reset_parameters()
        except Exception as e:
            logger.error(f"Initialization error in BayesianLinear: {e}")
            logger.error(traceback.format_exc())
            raise

    def _reset_parameters(self) -> None:
        """Initializes weight and bias variational parameters securely."""
        try:
            stdv = 1.0 / math.sqrt(self.weight_mu.size(1))
            self.weight_mu.data.uniform_(-stdv, stdv)
            # Initialize rho to yield a reasonable initial standard deviation via softplus
            self.weight_rho.data.fill_(-3.0)

            if self.bias_mu is not None:
                self.bias_mu.data.uniform_(-stdv, stdv)
                self.bias_rho.data.fill_(-3.0)
        except Exception as e:
            logger.error(f"Error during parameter reset in BayesianLinear: {e}")
            logger.error(traceback.format_exc())
            raise

    def forward(self, input: torch.Tensor, sample: bool = True) -> torch.Tensor:
        """
        Executes forward pass using the reparameterization trick to sample weights
        while maintaining backpropagation differentiability.
        """
        try:
            if sample:
                # Weight sampling via reparameterization trick: W = mu + sigma * epsilon
                weight_sigma = F.softplus(self.weight_rho)
                weight_epsilon = torch.randn_like(self.weight_mu)
                weight = self.weight_mu + weight_sigma * weight_epsilon

                # Bias sampling via reparameterization trick: b = mu + sigma * epsilon
                bias_sigma = F.softplus(self.bias_rho)
                bias_epsilon = torch.randn_like(self.bias_mu)
                bias = self.bias_mu + bias_sigma * bias_epsilon
            else:
                # Deterministic inference mode using posterior means
                weight = self.weight_mu
                bias = self.bias_mu

            return F.linear(input, weight, bias)
        except Exception as e:
            logger.error(f"Error during BayesianLinear forward pass: {e}")
            logger.error(traceback.format_exc())
            raise

    kl_divergence = property(lambda self: self._compute_kl_divergence())

    def _compute_kl_divergence(self) -> torch.Tensor:
        """
        Computes the analytical Kullback-Leibler (KL) Divergence between the
        variational Gaussian posterior and a standard Gaussian prior N(0, prior_std^2).
        """
        try:
            weight_sigma = F.softplus(self.weight_rho)
            bias_sigma = F.softplus(self.bias_rho)

            kl_weight = self._gaussian_kl(self.weight_mu, weight_sigma, self.prior_std)
            kl_bias = self._gaussian_kl(self.bias_mu, bias_sigma, self.prior_std)

            return kl_weight + kl_bias
        except Exception as e:
            logger.error(f"Error computing KL divergence in BayesianLinear: {e}")
            logger.error(traceback.format_exc())
            raise

    @staticmethod
    def _gaussian_kl(mu: torch.Tensor, sigma: torch.Tensor, prior_std: float) -> torch.Tensor:
        """Helper function calculating individual KL divergence components."""
        try:
            prior_var = prior_std ** 2
            var = sigma ** 2
            kl_term = torch.log(prior_std / sigma) + (var + mu ** 2) / (2 * prior_var) - 0.5
            return torch.sum(kl_term)
        except Exception as e:
            logger.error(f"Error in gaussian_kl calculation: {e}")
            logger.error(traceback.format_exc())
            raise


# =====================================================================
# Bayesian Neural Network Architecture
# =====================================================================
class BayesianNeuralNetwork(nn.Module):
    """
    Multi-layer Bayesian Neural Network constructed from custom BayesianLinear layers.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, prior_std: float = 1.0) -> None:
        super(BayesianNeuralNetwork, self).__init__()
        try:
            self.layer1 = BayesianLinear(input_dim, hidden_dim, prior_std=prior_std)
            self.layer2 = BayesianLinear(hidden_dim, hidden_dim, prior_std=prior_std)
            self.layer3 = BayesianLinear(hidden_dim, output_dim, prior_std=prior_std)
        except Exception as e:
            logger.error(f"Error instantiating BayesianNeuralNetwork: {e}")
            logger.error(traceback.format_exc())
            raise

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        try:
            x = F.relu(self.layer1(x, sample=sample))
            x = F.relu(self.layer2(x, sample=sample))
            x = self.layer3(x, sample=sample)
            return x
        except Exception as e:
            logger.error(f"Error during BNN forward pass: {e}")
            logger.error(traceback.format_exc())
            raise

    def kl_divergence(self) -> torch.Tensor:
        """Aggregates structural KL divergence across all variational network layers."""
        try:
            return self.layer1.kl_divergence + self.layer2.kl_divergence + self.layer3.kl_divergence
        except Exception as e:
            logger.error(f"Error accumulating network KL divergence: {e}")
            logger.error(traceback.format_exc())
            raise

    def sample_predict(self, x: torch.Tensor, num_samples: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes multiple Monte Carlo forward passes to quantify predictive uncertainty
        (evaluating both epistemic and aleatoric manifestations).
        """
        try:
            preds = []
            for _ in range(num_samples):
                preds.append(self.forward(x, sample=True))
            
            preds_stack = torch.stack(preds, dim=0) # Shape: [num_samples, batch_size, output_dim]
            mean_pred = torch.mean(preds_stack, dim=0)
            std_pred = torch.std(preds_stack, dim=0)
            return mean_pred, std_pred
        except Exception as e:
            logger.error(f"Error during Monte Carlo sample prediction: {e}")
            logger.error(traceback.format_exc())
            raise


# =====================================================================
# ELBO Optimization Engine Loss Function
# =====================================================================
class ELBOLoss(nn.Module):
    """
    Evidence Lower Bound (ELBO) optimization objective consisting of
    Negative Data Likelihood (MSE for regression) + Scaled KL Divergence penalty.
    """
    def __init__(self, num_batches: int) -> None:
        super(ELBOLoss, self).__init__()
        self.num_batches = num_batches

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor, kl_div: torch.Tensor) -> torch.Tensor:
        try:
            # Data likelihood loss (Negative Log-Likelihood proxied via Mean Squared Error sum)
            nll_loss = F.mse_loss(outputs, targets, reduction='sum')
            
            # Scale structural KL penalty inversely proportional to batch count (Blundell et al.)
            scaled_kl = kl_div / float(self.num_batches)

            elbo = nll_loss + scaled_kl
            return elbo
        except Exception as e:
            logger.error(f"Error calculating ELBO loss objective: {e}")
            logger.error(traceback.format_exc())
            raise


# =====================================================================
# Synthetic Data Generation & Execution Pipeline
# =====================================================================
def generate_synthetic_regression_data(num_samples: int = 500) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generates non-linear synthetic regression data (noisy sine wave)."""
    try:
        np.random.seed(42)
        torch.manual_seed(42)
        
        x = np.random.uniform(-3.0, 3.0, num_samples)
        noise = np.random.normal(0, 0.15, num_samples)
        y = x * np.sin(x) + noise

        x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        return x_tensor, y_tensor
    except Exception as e:
        logger.error(f"Error generating synthetic dataset: {e}")
        logger.error(traceback.format_exc())
        raise


def main() -> None:
    """Main execution entry point demonstrating end-to-end BNN training and uncertainty estimation."""
    try:
        logger.info("Initializing Bayesian Neural Network production pipeline...")

        # Hyperparameters & Setup
        input_dim = 1
        hidden_dim = 32
        output_dim = 1
        prior_std = 0.5
        learning_rate = 0.005
        epochs = 800
        batch_size = 64

        # Generate dataset
        x_data, y_data = generate_synthetic_regression_data(num_samples=500)
        dataset = torch.utils.data.TensorDataset(x_data, y_data)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        num_batches = len(dataloader)

        # Initialize model, optimizer, and loss criterion
        model = BayesianNeuralNetwork(input_dim, hidden_dim, output_dim, prior_std=prior_std)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = ELBOLoss(num_batches=num_batches)

        # Training Optimization Loop
        logger.info("Commencing Variational Inference optimization loop...")
        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in dataloader:
                try:
                    optimizer.zero_grad()
                    
                    # Forward pass with Monte Carlo sampling
                    outputs = model(batch_x, sample=True)
                    kl_div = model.kl_divergence()
                    
                    loss = criterion(outputs, batch_y, kl_div)
                    
                    # Backpropagation and optimization step
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                except Exception as batch_error:
                    logger.error(f"Error encountered inside training batch loop at epoch {epoch+1}: {batch_error}")
                    logger.error(traceback.format_exc())
                    raise batch_error

            if (epoch + 1) % 100 == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}] | Negative ELBO Loss: {epoch_loss / num_batches:.4f}")

        logger.info("Training completed successfully. Executing Monte Carlo uncertainty inference...")
        model.eval()

        # Generate test inputs over a dense grid
        x_test = np.linspace(-4.0, 4.0, 200)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            mean_pred, std_pred = model.sample_predict(x_test_tensor, num_samples=50)

        logger.info("Inference completed successfully.")
        logger.info(f"Test Grid Shape: {x_test_tensor.shape}")
        logger.info(f"Predicted Mean Shape: {mean_pred.shape}")
        logger.info(f"Predicted Uncertainty (Std) Shape: {std_pred.shape}")
        logger.info("Sample uncertainty statistics -> Mean Std: %.4f, Max Std: %.4f", std_pred.mean().item(), std_pred.max().item())

    except Exception as e:
        logger.error(f"Critical execution failure in main pipeline: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()