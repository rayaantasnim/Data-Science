# Author: Rayaan tasnim, 13 aged prodigy

import os
import sys
import json
import logging
import traceback
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import scipy.stats as stats
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from pydantic import BaseModel, Field

from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = declarative_logger = logging.getLogger("SyntheticDataPipeline")

# ==========================================
# 1. Pydantic Models for Schema & Constraints
# ==========================================

class ColumnConstraint(BaseModel):
    column_name: str
    data_type: str = Field(..., description="float, int, or categorical")
    mean: Optional[float] = None
    std: Optional[float] = None
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    categories: Optional[List[str]] = None
    probabilities: Optional[List[float]] = None

class DatasetSchema(BaseModel):
    dataset_name: str
    description: str
    columns: List[ColumnConstraint]
    correlation_matrix: List[List[float]] = Field(..., description="NxN correlation matrix for numeric columns")
    conditional_rules: List[str] = Field(..., description="Logical rules e.g., 'if income > 50000 then credit_score variance increases'")

class PipelineConfig(BaseModel):
    target_rows: int = 10000
    random_seed: int = 42


# ==========================================
# 2. LLM Schema Bootstrapping Engine
# ==========================================

class SchemaEngine:
    def __init__(self, api_key: Optional[str] = None):
        try:
            resolved_key = api_key or os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY")
            self.client = OpenAI(api_key=resolved_key)
            logger.info("OpenAI client initialized successfully for Schema Engine.")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            traceback.print_exc()
            raise

    def bootstrap_schema(self, metadata_description: str) -> DatasetSchema:
        """
        Uses OpenAI to analyze metadata description and output structured schema,
        distributions, correlation boundaries, and conditional rules.
        """
        try:
            logger.info("Bootstrapping dataset schema via LLM...")
            prompt = (
                f"Analyze the following dataset metadata and design a rigorous statistical schema, "
                f"including column distributions, a correlation matrix for numeric columns, and conditional rules.\n\n"
                f"Metadata Description:\n{metadata_description}"
            )

            response = self.client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a principal data architect and statistical modelling expert."},
                    {"role": "user", "content": prompt}
                ],
                response_format=DatasetSchema,
            )
            schema = response.choices.message.parsed
            logger.info(f"Schema successfully bootstrapped for dataset: {schema.dataset_name}")
            return schema
        except Exception as e:
            logger.error(f"Error during LLM schema bootstrapping: {e}")
            traceback.print_exc()
            # Fallback mock schema for robustness if API call fails or mock key is used
            logger.warning("Falling back to default synthetic schema for demonstration/testing.")
            return DatasetSchema(
                dataset_name="DefaultCustomerLoanDataset",
                description="Fallback schema for credit risk and customer analytics.",
                columns=[
                    ColumnConstraint(column_name="age", data_type="float", mean=38.5, std=12.0, min_val=18.0, max_val=85.0),
                    ColumnConstraint(column_name="income", data_type="float", mean=65000.0, std=25000.0, min_val=15000.0, max_val=200000.0),
                    ColumnConstraint(column_name="debt_ratio", data_type="float", mean=0.35, std=0.15, min_val=0.0, max_val=1.0),
                    ColumnConstraint(column_name="default_status", data_type="categorical", categories=["0", "1"], probabilities=[0.85, 0.15])
                ],
                correlation_matrix=[
                    [1.00,  0.35, -0.10, -0.20],
                    [0.35,  1.00, -0.05, -0.45],
                    [-0.10, -0.05, 1.00,  0.55],
                    [-0.20, -0.45,  0.55,  1.00]
                ],
                conditional_rules=[
                    "if income < 30000 then debt_ratio variance scales by 1.5x",
                    "if debt_ratio > 0.6 then default_status probability of '1' increases to 0.60"
                ]
            )


# ==========================================
# 3. High-Throughput Probabilistic Co-Simulation Module
# ==========================================

class ProbabilisticSimulator:
    def __init__(self, schema: DatasetSchema, config: PipelineConfig):
        self.schema = schema
        self.config = config
        np.random.seed(config.random_seed)

    def generate_population(self) -> Tuple[np.ndarray, List[str]]:
        """
        Generates high-throughput synthetic data using Multivariate Normal distributions,
        Cholesky decomposition for correlation enforcement, and applies conditional dependencies.
        """
        try:
            logger.info(f"Starting high-throughput simulation for {self.config.target_rows} rows...")
            numeric_cols = [c for c in self.schema.columns if c.data_type in ["float", "int"]]
            categorical_cols = [c for c in self.schema.columns if c.data_type == "categorical"]

            n_samples = self.config.target_rows
            n_numeric = len(numeric_cols)

            # Build mean and covariance vectors
            means = np.array([col.mean if col.mean is not None else 0.0 for col in numeric_cols])
            stds = np.array([col.std if col.std is not None else 1.0 for col in numeric_cols])
            
            # Construct covariance matrix from correlation matrix and standard deviations
            corr_matrix = np.array(self.schema.correlation_matrix)
            if corr_matrix.shape != (n_numeric, n_numeric):
                logger.warning("Provided correlation matrix shape mismatch. Using identity matrix.")
                corr_matrix = np.eye(n_numeric)
            
            cov_matrix = np.outer(stds, stds) * corr_matrix

            # Sample from Multivariate Normal distribution
            logger.info("Sampling from Multivariate Normal distribution...")
            raw_numeric_data = np.random.multivariate_normal(mean=means, cov=cov_matrix, size=n_samples)

            # Apply min/max clipping constraints
            for idx, col in enumerate(numeric_cols):
                if col.min_val is not None:
                    raw_numeric_data[:, idx] = np.maximum(raw_numeric_data[:, idx], col.min_val)
                if col.max_val is not None:
                    raw_numeric_data[:, idx] = np.minimum(raw_numeric_data[:, idx], col.max_val)

            # Apply conditional rules and dependency injection
            # Example rule implementation: "if income < 30000 then debt_ratio variance scales"
            income_idx = next((i for i, c in enumerate(numeric_cols) if c.column_name == "income"), None)
            debt_idx = next((i for i, c in enumerate(numeric_cols) if c.column_name == "debt_ratio"), None)
            
            if income_idx is not None and debt_idx is not None:
                low_income_mask = raw_numeric_data[:, income_idx] < 35000.0
                raw_numeric_data[low_income_mask, debt_idx] *= np.random.uniform(1.2, 1.8, size=np.sum(low_income_mask))
                raw_numeric_data[:, debt_idx] = np.clip(raw_numeric_data[:, debt_idx], 0.0, 1.0)

            # Generate categorical columns based on conditional probabilities or baseline multinomial sampling
            categorical_data = []
            for col in categorical_cols:
                probs = col.probabilities or [0.5, 0.5]
                categories = col.categories or ["0", "1"]
                
                # Dynamic adjustment based on debt_ratio if applicable
                if col.column_name == "default_status" and debt_idx is not None:
                    # Higher debt ratio increases probability of default ('1')
                    debt_values = raw_numeric_data[:, debt_idx]
                    # Sigmoid adjustment for dynamic probability
                    p_default = 1.0 / (1.0 + np.exp(-10 * (debt_values - 0.5)))
                    choices = np.array([np.random.choice(categories, p=[1 - p, p]) for p in p_default])
                    categorical_data.append(choices.reshape(-1, 1))
                else:
                    choices = np.random.choice(categories, size=n_samples, p=probs)
                    categorical_data.append(choices.reshape(-1, 1))

            if categorical_data:
                cat_array = np.hstack(categorical_data)
                full_dataset = np.hstack((raw_numeric_data, cat_array))
            else:
                full_dataset = raw_numeric_data

            column_names = [c.column_name for c in numeric_cols] + [c.column_name for c in categorical_cols]
            logger.info("Probabilistic co-simulation completed successfully.")
            return full_dataset, column_names

        except Exception as e:
            logger.error(f"Error during probabilistic co-simulation: {e}")
            traceback.print_exc()
            raise


# ==========================================
# 4. Automated Downstream Utility & Fidelity Auditor
# ==========================================

class FidelityAuditor:
    def __init__(self, real_data: np.ndarray, synthetic_data: np.ndarray, column_names: List[str]):
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.column_names = column_names

    def compute_statistical_metrics(self) -> Dict[str, Any]:
        """
        Computes Wasserstein Distance, Jensen-Shannon Divergence, and Covariance Matrix Error
        completely from scratch using NumPy and SciPy.
        """
        try:
            logger.info("Computing statistical fidelity metrics...")
            metrics = {}
            
            # Numeric columns evaluation (assume all except last if categorical, or parse by type)
            # For demonstration, assume numeric are all columns except the last categorical one
            n_numeric = self.real_data.shape[1] - 1 if len(self.column_names) > 3 else self.real_data.shape[1]

            wasserstein_distances = {}
            js_divergences = {}

            for i in range(n_numeric):
                col_name = self.column_names[i]
                real_col = self.real_data[:, i].astype(float)
                synth_col = self.synthetic_data[:, i].astype(float)

                # Wasserstein Distance
                w_dist = stats.wasserstein_distance(real_col, synth_col)
                wasserstein_distances[col_name] = float(w_dist)

                # Jensen-Shannon Divergence (using binned probability distributions)
                min_val = min(real_col.min(), synth_col.min())
                max_val = max(real_col.max(), synth_col.max())
                bins = np.linspace(min_val, max_val, 50)
                
                p, _ = np.histogram(real_col, bins=bins, density=True)
                q, _ = np.histogram(synth_col, bins=bins, density=True)
                
                # Normalize to sum to 1
                p = p + 1e-8; p /= np.sum(p)
                q = q + 1e-8; q /= np.sum(q)
                
                js_div = jensenshannon(p, q)
                js_divergences[col_name] = float(js_div)

            # Covariance Matrix Error (Frobenius norm of difference between covariance matrices)
            real_cov = np.cov(self.real_data[:, :n_numeric].astype(float), rowvar=False)
            synth_cov = np.cov(self.synthetic_data[:, :n_numeric].astype(float), rowvar=False)
            cov_error = float(np.linalg.norm(real_cov - synth_cov, 'fro'))

            metrics["wasserstein_distances"] = wasserstein_distances
            metrics["jensen_shannon_divergence"] = js_divergences
            metrics["covariance_matrix_frobenius_error"] = cov_error

            logger.info("Statistical metrics computed successfully.")
            return metrics

        except Exception as e:
            logger.error(f"Error computing statistical metrics: {e}")
            traceback.print_exc()
            return {"error": str(e)}

    def evaluate_tstr_utility(self) -> Dict[str, float]:
        """
        Runs 'Train Synthetic, Test Real' (TSTR) evaluation using scikit-learn Logistic Regression.
        Trains a classifier on synthetic data and evaluates performance on real holdout test data.
        """
        try:
            logger.info("Executing Train Synthetic, Test Real (TSTR) utility evaluation...")
            
            # Prepare features and target (assume last column is binary target)
            n_cols = self.real_data.shape[1]
            X_real = self.real_data[:, :-1].astype(float)
            y_real = self.real_data[:, -1].astype(int)

            X_synth = self.synthetic_data[:, :-1].astype(float)
            y_synth = self.synthetic_data[:, -1].astype(int)

            # Split real data for testing evaluation
            _, X_test, _, y_test = train_test_split(X_real, y_real, test_size=0.3, random_state=42)

            # Train classifier on SYNTHETIC data
            clf = LogisticRegression(max_iter=1000, random_state=42)
            clf.fit(X_synth, y_synth)

            # Evaluate on REAL test data
            y_pred = clf.predict(X_test)
            y_proba = clf.predict_proba(X_test)[:, 1]

            acc = accuracy_score(y_test, y_pred)
            try:
                auc = roc_auc_score(y_test, y_proba)
            except Exception:
                auc = 0.5

            logger.info(f"TSTR Evaluation Results -> Accuracy: {acc:.4f}, AUC: {auc:.4f}")
            return {"tstr_accuracy": float(acc), "tstr_roc_auc": float(auc)}

        except Exception as e:
            logger.error(f"Error during TSTR evaluation: {e}")
            traceback.print_exc()
            return {"tstr_accuracy": 0.0, "tstr_roc_auc": 0.0}


# ==========================================
# 5. Main Execution Orchestration
# ==========================================

if __name__ == "__main__":
    try:
        logger.info("Initializing End-to-End LLM-Guided Synthetic Data Generation Pipeline...")

        # Step 1: Configuration & Metadata Definition
        config = PipelineConfig(target_rows=5000, random_seed=42)
        metadata_prompt = (
            "Dataset representing retail banking customer credit loans containing customer age, "
            "annual income, debt-to-income ratio, and default status (binary 0 or 1). "
            "Income correlates positively with age and negatively with debt ratio and default."
        )

        # Step 2: Schema Bootstrapping
        schema_engine = SchemaEngine()
        schema = schema_engine.bootstrap_schema(metadata_prompt)

        # Step 3: Probabilistic Co-Simulation
        simulator = ProbabilisticSimulator(schema, config)
        synthetic_population, column_names = simulator.generate_population()

        # Generate a simulated 'real' baseline dataset for fidelity auditing comparison
        logger.info("Generating baseline real-world sample dataset for comparison...")
        np.random.seed(100)
        real_baseline_numeric = np.random.multivariate_normal(
            mean=[37.0, 64000.0, 0.33],
            cov=[[11.0, 24000.0, -0.09], [24000.0, 2400000000.0, -0.04], [-0.09, -0.04, 0.02]],
            size=2000
        )
        real_baseline_numeric[:, 0] = np.clip(real_baseline_numeric[:, 0], 18, 85)
        real_baseline_numeric[:, 1] = np.clip(real_baseline_numeric[:, 1], 15000, 200000)
        real_baseline_numeric[:, 2] = np.clip(real_baseline_numeric[:, 2], 0.0, 1.0)
        
        real_default_status = np.random.choice(["0", "1"], size=2000, p=[0.88, 0.12]).reshape(-1, 1)
        real_dataset = np.hstack((real_baseline_numeric, real_default_status))

        # Step 4: Fidelity & Utility Auditing
        auditor = FidelityAuditor(real_data=real_dataset, synthetic_data=synthetic_population, column_names=column_names)
        
        fidelity_metrics = auditor.computed_statistical_metrics() if hasattr(auditor, 'compute_statistical_metrics') else auditor.compute_statistical_metrics()
        tstr_results = auditor.evaluate_tstr_utility()

        # Step 5: Output and Summary Reporting
        report = {
            "dataset_name": schema.dataset_name,
            "total_synthetic_rows_generated": config.target_rows,
            "fidelity_metrics": fidelity_metrics,
            "utility_metrics": tstr_results
        }

        print("\n" + "="*50)
        print("PIPELINE EXECUTION SUMMARY REPORT")
        print("="*50)
        print(json.dumps(report, indent=4))
        print("="*50)
        logger.info("Pipeline executed successfully from end to end.")

    except Exception as e:
        logger.critical(f"Critical failure in pipeline execution: {e}")
        traceback.print_exc()
        sys.exit(1)