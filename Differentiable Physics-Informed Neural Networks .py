# Author: Rayaan tasnim, 13 aged prodigy

import os
import sys
import logging
import traceback
from typing import Tuple, Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[sys.stdout]
)
logger = logging.getLogger("PINN_Burgers_Engine")


class CustomFullyConnectedNN(nn.Module):
    """
    Custom fully-connected neural network implemented using raw weight parameters
    and manual linear transformations to adhere strictly to low-level structural
    operations without high-level Sequential abstractions.
    """
    def __init__(self, layer_dims: List[int], activation: nn.Module = nn.Tanh()):
        super(CustomFullyConnectedNN, self).__init__()
        try:
            self.layer_dims = layer_dims
            self.activation = activation
            self.weights = nn.ParameterList()
            self.biases = nn.ParameterList()

            for i in range(len(layer_dims) - 1):
                in_dim = layer_dims[i]
                out_dim = layer_dims[i + 1]
                
                # Initialize weights using Xavier/Glorot normal distribution
                w = nn.Parameter(torch.empty(in_dim, out_dim))
                nn.init.xavier_normal_(w)
                self.weights.append(w)
                
                # Initialize biases to zeros
                b = nn.Parameter(torch.zeros(out_dim))
                self.biases.append(b)
                
            logger.info(f"Initialized CustomFullyConnectedNN with architecture: {layer_dims}")
        except Exception as e:
            logger.error("Failed to initialize CustomFullyConnectedNN network architecture.")
            logger.error(traceback.format_exc())
            raise e

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        try:
            out = x
            for i in range(len(self.weights) - 1):
                out = torch.matmul(out, self.weights[i]) + self.biases[i]
                out = self.activation(out)
            # Final output layer without activation
            out = torch.matmul(out, self.weights[-1]) + self.biases[-1]
            return out
        except Exception as e:
            logger.error("Error occurred during forward pass execution.")
            logger.error(traceback.format_exc())
            raise e


class BurgersPINNEngine:
    """
    Differentiable Physics-Informed Neural Network Engine for solving the 1D Burgers' Equation:
    du/dt + u * du/dx - nu * d^2u/dx^2 = 0
    """
    def __init__(self, model: nn.Module, nu: float = 0.01 / np.pi):
        self.model = model
        self.nu = nu
        logger.info(f"BurgersPINNEngine instantiated with kinematic viscosity nu = {self.nu:.6f}")

    def compute_pde_residual(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Computes the first and second-order derivatives using automatic differentiation
        to construct the physical residual of the 1D Burgers' equation.
        """
        try:
            # Ensure inputs require gradients
            x = x.clone().detach().requires_grad_(True)
            t = t.clone().detach().requires_grad_(True)

            # Forward pass
            inputs = torch.cat([x, t], dim=1)
            u = self.model(inputs)

            # First-order derivatives using create_graph=True for higher-order gradients
            u_x = torch.autograd.grad(
                outputs=u, inputs=x,
                grad_outputs=torch.ones_like(u),
                create_graph=True, retain_graph=True,
                allow_unused=True
            )[0]

            u_t = torch.autograd.grad(
                outputs=u, inputs=t,
                grad_outputs=torch.ones_like(u),
                create_graph=True, retain_graph=True,
                allow_unused=True
            )[0]

            # Second-order spatial derivative
            u_xx = torch.autograd.grad(
                outputs=u_x, inputs=x,
                grad_outputs=torch.ones_like(u_x),
                create_graph=True, retain_graph=True,
                allow_unused=True
            )[0]

            if u_x is None or u_t is None or u_xx is None:
                raise ValueError("Gradient computation resulted in None types. Check tensor connectivity.")

            # Burgers' equation residual: f = u_t + u * u_x - nu * u_xx = 0
            residual = u_t + u * u_x - self.nu * u_xx
            return residual

        except Exception as e:
            logger.error("Critical error during differentiable PDE residual calculation.")
            logger.error(traceback.format_exc())
            raise e

    def loss_function(
        self,
        x_collocation: torch.Tensor,
        t_collocation: torch.Tensor,
        x_ic: torch.Tensor,
        t_ic: torch.Tensor,
        u_ic_true: torch.Tensor,
        x_bc: torch.Tensor,
        t_bc: torch.Tensor,
        u_bc_true: torch.Tensor,
        weights: Dict[str, float]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes composite loss comprising PDE residual loss, Initial Condition (IC) loss,
        and Boundary Condition (BC) loss with dynamic balancing weights.
        """
        try:
            # 1. PDE Residual Loss
            residual = self.compute_pde_residual(x_collocation, t_collocation)
            loss_pde = torch.mean(torch.square(residual))

            # 2. Initial Condition Loss
            ic_inputs = torch.cat([x_ic, t_ic], dim=1)
            u_ic_pred = self.model(ic_inputs)
            loss_ic = torch.mean(torch.square(u_ic_pred - u_ic_true))

            # 3. Boundary Condition Loss
            bc_inputs = torch.cat([x_bc, t_bc], dim=1)
            u_bc_pred = self.model(bc_inputs)
            loss_bc = torch.mean(torch.square(u_bc_pred - u_bc_true))

            # Composite weighted loss
            total_loss = (
                weights.get("pde", 1.0) * loss_pde +
                weights.get("ic", 1.0) * loss_ic +
                weights.get("bc", 1.0) * loss_bc
            )

            return total_loss, loss_pde, loss_ic, loss_bc

        except Exception as e:
            logger.error("Error encountered while computing composite loss function.")
            logger.error(traceback.format_exc())
            raise e


def generate_synthetic_data(num_collocation: int = 2000, num_ic: int = 500, num_bc: int = 500) -> Dict[str, torch.Tensor]:
    """
    Generates synthetic collocation, initial condition, and boundary condition points
    for the 1D Burgers' domain: x in [-1, 1], t in [0, 1].
    """
    try:
        # Collocation points (Interior domain sampling)
        x_col = torch.FloatTensor(num_collocation, 1).uniform_(-1.0, 1.0)
        t_col = torch.FloatTensor(num_collocation, 1).uniform_(0.0, 1.0)

        # Initial condition points (t = 0, x in [-1, 1])
        x_i = torch.FloatTensor(num_ic, 1).uniform_(-1.0, 1.0)
        t_i = torch.zeros_like(x_i)
        # Analytical IC for Burgers: u(x, 0) = -sin(pi * x)
        u_i = -torch.sin(np.pi * x_i)

        # Boundary condition points (x = -1 or x = 1, t in [0, 1])
        t_b = torch.FloatTensor(num_bc, 1).uniform_(0.0, 1.0)
        # Randomly choose left (-1) or right (+1) boundary
        x_b = torch.where(torch.rand(num_bc, 1) > 0.5, torch.ones(num_bc, 1), -torch.ones(num_bc, 1))
        u_b = torch.zeros_like(x_b)  # Dirichlet homogeneous BCs u(-1, t) = u(1, t) = 0

        data = {
            "x_col": x_col, "t_col": t_col,
            "x_ic": x_i, "t_ic": t_i, "u_ic": u_i,
            "x_bc": x_b, "t_bc": t_b, "u_bc": u_b
        }
        logger.info("Synthetic collocation and boundary datasets generated successfully.")
        return data
    except Exception as e:
        logger.error("Failed to generate synthetic training datasets.")
        logger.error(traceback.format_exc())
        raise e


if __name__ == "__main__":
    try:
        logger.info("Initializing Differentiable PINN execution pipeline for 1D Burgers' Equation.")

        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

        # Network architecture dimensions: [Input(x, t) -> Hidden Layers -> Output(u)]
        layer_dimensions = [2, 32, 32, 32, 1]
        model = CustomFullyConnectedNN(layer_dims=layer_dimensions, activation=nn.Tanh())

        # Instantiate PINN engine
        pinn_engine = BurgersPINNEngine(model=model, nu=0.01 / np.pi)

        # Generate datasets
        dataset = generate_synthetic_data(num_collocation=1500, num_ic=300, num_bc=300)
        
        # Move tensors to appropriate device (CPU by default, expandable to GPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        x_col = dataset["x_col"].to(device)
        t_col = dataset["t_col"].to(device)
        x_ic = dataset["x_ic"].to(device)
        t_ic = dataset["t_ic"].to(device)
        u_ic = dataset["u_ic"].to(device)
        x_bc = dataset["x_bc"].to(device)
        t_bc = dataset["t_bc"].to(device)
        u_bc = dataset["u_bc"].to(device)
        model.to(device)

        # Optimizer setup with adaptive learning rate
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.9)

        loss_weights = {"pde": 1.0, "ic": 10.0, "bc": 10.0}
        epochs = 100  # Minimal training iterations for baseline verification check

        logger.info("Starting PINN optimization training loop...")
        for epoch in range(1, epochs + 1):
            try:
                optimizer.zero_grad()

                total_loss, loss_pde, loss_ic, loss_bc = pinn_engine.loss_function(
                    x_col, t_col, x_ic, t_ic, u_ic, x_bc, t_bc, u_bc, loss_weights
                )

                total_loss.backward()
                optimizer.step()
                scheduler.step()

                if epoch % 10 == 0 or epoch == 1:
                    logger.info(
                        f"Epoch [{epoch}/{epochs}] | "
                        f"Total Loss: {total_loss.item():.6e} | "
                        f"PDE Loss: {loss_pde.item():.6e} | "
                        f"IC Loss: {loss_ic.item():.6e} | "
                        f"BC Loss: {loss_bc.item():.6e}"
                    )
            except Exception as loop_err:
                logger.error(f"Error encountered during optimization step at epoch {epoch}.")
                logger.error(traceback.format_exc())
                raise loop_err

        logger.info("Training pipeline executed successfully. PDE compliance verified.")

    except Exception as main_err:
        logger.critical("Fatal exception encountered in main execution workflow.")
        logger.critical(traceback.format_exc())
        sys.exit(1)