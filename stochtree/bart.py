import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold
from typing import Optional
from scipy.linalg import lstsq
from scipy.stats import gamma
from .data import Dataset, Residual
from .forest import ForestContainer
from .sampler import ForestSampler, RNG, GlobalVarianceModel, LeafVarianceModel
from .utils import NotSampledError

class BARTModel:
    def __init__(self) -> None:
        # Internal flag for whether the sample() method has been run
        self.sampled = False
        self.rng = np.random.default_rng()
    
    def is_sampled(self) -> bool:
        return self.sampled
    
    def sample(self, X_train: np.array, y_train: np.array, basis_train: np.array = None, X_test: np.array = None, basis_test: np.array = None, 
               feature_types: np.array = None, cutpoint_grid_size = 100, sigma_leaf: float = None, alpha: float = 0.95, beta: float = 2.0, 
               min_samples_leaf: int = 5, nu: float = 3, lamb: float = None, a_leaf: float = 3, b_leaf: float = None, q: float = 0.9, 
               sigma2: float = None, num_trees: int = 200, num_gfr: int = 5, num_burnin: int = 0, num_mcmc: int = 100, 
               sample_sigma_global: bool = True, sample_sigma_leaf: bool = True, random_seed: int = -1) -> None:
        # Convert everything to standard shape (2-dimensional)
        if X_train.ndim == 1:
            X_train = np.expand_dims(X_train, 1)
        if basis_train is not None:
            if basis_train.ndim == 1:
                basis_train = np.expand_dims(basis_train, 1)
        if y_train.ndim == 1:
            y_train = np.expand_dims(y_train, 1)
        if X_test is not None:
            if X_train.ndim == 1:
                X_train = np.expand_dims(X_train, 1)
        if basis_test is not None:
            if basis_test.ndim == 1:
                basis_test = np.expand_dims(basis_test, 1)
        
        # Data checks
        if X_test is not None:
            if X_test.shape[1] != X_train.shape[1]:
                raise ValueError("X_train and X_test must have the same number of columns")
        if basis_test is not None:
            if basis_train is not None:
                if basis_test.shape[1] != basis_train.shape[1]:
                    raise ValueError("basis_train and basis_test must have the same number of columns")
            else:
                raise ValueError("basis_test provided but basis_train was not")
        if basis_train is not None:
            if basis_train.shape[0] != X_train.shape[0]:
                raise ValueError("basis_train and Z_train must have the same number of rows")
        if y_train.shape[0] != X_train.shape[0]:
            raise ValueError("X_train and y_train must have the same number of rows")
        if X_test is not None and basis_test is not None:
            if X_test.shape[0] != basis_test.shape[0]:
                raise ValueError("X_test and basis_test must have the same number of rows")

        # Determine whether a test set is provided
        self.has_test = X_test is not None

        # Determine whether a basis is provided
        self.has_basis = basis_train is not None

        # Unpack data dimensions
        self.n_train = y_train.shape[0]
        self.n_test = X_test.shape[0] if self.has_test else 0
        self.num_covariates = X_train.shape[1]
        self.num_basis = basis_train.shape[1] if self.has_basis else 0

        # Set feature type defaults if not provided
        if feature_types is None:
            feature_types = np.zeros(self.num_covariates)
        
        # Set variable weights for the prognostic and treatment effect forests
        variable_weights = np.repeat(1.0/X_train.shape[1], X_train.shape[1])

        # Scale outcome
        self.y_bar = np.squeeze(np.mean(y_train))
        self.y_std = np.squeeze(np.std(y_train))
        resid_train = (y_train-self.y_bar)/self.y_std

        # Calibrate priors for global sigma^2 and sigma_leaf
        if lamb is None:
            reg_basis = np.c_[np.ones(self.n_train),X_train]
            reg_soln = lstsq(reg_basis, np.squeeze(resid_train))
            sigma2hat = reg_soln[1] / self.n_train
            quantile_cutoff = 0.9
            lamb = (sigma2hat*gamma.ppf(1-quantile_cutoff,nu))/nu
        sigma2 = sigma2hat if sigma2 is None else sigma2
        b_leaf = np.squeeze(np.var(resid_train)) / num_trees if b_leaf is None else b_leaf
        sigma_leaf = np.squeeze(np.var(resid_train)) / num_trees if sigma_leaf is None else sigma_leaf
        current_sigma2 = sigma2
        current_leaf_scale = np.array([[sigma_leaf]])

        # Container of variance parameter samples
        self.num_gfr = num_gfr
        self.num_burnin = num_burnin
        self.num_mcmc = num_mcmc
        self.num_samples = num_gfr + num_burnin + num_mcmc
        self.sample_sigma_global = sample_sigma_global
        self.sample_sigma_leaf = sample_sigma_leaf
        if sample_sigma_global:
            self.global_var_samples = np.zeros(self.num_samples)
        if sample_sigma_leaf:
            self.leaf_scale_samples = np.zeros(self.num_samples)
        
        # Forest Dataset (covariates and optional basis)
        forest_dataset_train = Dataset()
        forest_dataset_train.add_covariates(X_train)
        if self.has_basis:
            forest_dataset_train.add_basis(basis_train)
        if self.has_test:
            forest_dataset_test = Dataset()
            forest_dataset_test.add_covariates(X_test)
            if self.has_basis:
                forest_dataset_test.add_basis(basis_test)

        # Residual
        residual_train = Residual(resid_train)

        # C++ random number generator
        if random_seed is None: 
            cpp_rng = RNG(-1)
        else:
            cpp_rng = RNG(random_seed)
        
        # Sampling data structures
        forest_sampler = ForestSampler(forest_dataset_train, feature_types, num_trees, self.n_train, alpha, beta, min_samples_leaf)

        # Determine the leaf model
        if not self.has_basis:
            leaf_model_int = 0
        elif self.num_basis == 1:
            leaf_model_int = 1
        else:
            leaf_model_int = 2
        
        # Container of forest samples
        self.forest_container = ForestContainer(num_trees, 1, True) if not self.has_basis else ForestContainer(num_trees, self.num_basis, False)
        
        # Variance samplers
        if self.sample_sigma_global:
            global_var_model = GlobalVarianceModel()
        if self.sample_sigma_leaf:
            leaf_var_model = LeafVarianceModel()

        # Initialize the leaves of each tree in the prognostic forest
        init_root = np.squeeze(np.mean(resid_train)) / num_trees
        self.forest_container.set_root_leaves(0, init_root)
        forest_sampler.update_residual(forest_dataset_train, residual_train, self.forest_container, False, 0, True)

        # Run GFR (warm start) if specified
        if self.num_gfr > 0:
            for i in range(self.num_gfr):
                # Sample the forest
                forest_sampler.sample_one_iteration(
                    self.forest_container, forest_dataset_train, residual_train, cpp_rng, feature_types, 
                    cutpoint_grid_size, current_leaf_scale, variable_weights, current_sigma2, leaf_model_int, True, True
                )

                # Sample variance parameters (if requested)
                if self.sample_sigma_global:
                    current_sigma2 = global_var_model.sample_one_iteration(residual_train, cpp_rng, nu, lamb)
                    self.global_var_samples[i] = current_sigma2*self.y_std*self.y_std
                if self.sample_sigma_leaf:
                    self.leaf_scale_samples[i] = leaf_var_model.sample_one_iteration(self.forest_container, cpp_rng, a_leaf, b_leaf, i)
                    current_leaf_scale[0,0] = self.leaf_scale_samples[i]
        
        # Run MCMC
        if self.num_burnin + self.num_mcmc > 0:
            for i in range(self.num_gfr, self.num_samples):
                # Sample the forest
                forest_sampler.sample_one_iteration(
                    self.forest_container, forest_dataset_train, residual_train, cpp_rng, feature_types, 
                    cutpoint_grid_size, current_leaf_scale, variable_weights, current_sigma2, leaf_model_int, False, True
                )

                # Sample variance parameters (if requested)
                if self.sample_sigma_global:
                    current_sigma2 = global_var_model.sample_one_iteration(residual_train, cpp_rng, nu, lamb)
                    self.global_var_samples[i] = current_sigma2*self.y_std*self.y_std
                if self.sample_sigma_leaf:
                    self.leaf_scale_samples[i] = leaf_var_model.sample_one_iteration(self.forest_container, cpp_rng, a_leaf, b_leaf, i)
                    current_leaf_scale[0,0] = self.leaf_scale_samples[i]
        
        # Mark the model as sampled
        self.sampled = True
        
        # Store predictions
        yhat_train_raw = self.forest_container.forest_container_cpp.Predict(forest_dataset_train.dataset_cpp)
        self.y_hat_train = yhat_train_raw*self.y_std + self.y_bar
        if self.has_test:
            yhat_test_raw = self.forest_container.forest_container_cpp.Predict(forest_dataset_test.dataset_cpp)
            self.y_hat_test = yhat_test_raw*self.y_std + self.y_bar
    
    def predict(self, covariates: np.array, basis: np.array = None) -> np.array:
        if not self.is_sampled():
            msg = (
                "This BCFModel instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this model."
            )
            raise NotSampledError(msg)
        
        # Convert everything to standard shape (2-dimensional)
        if covariates.ndim == 1:
            covariates = np.expand_dims(covariates, 1)
        if basis.ndim == 1:
            basis = np.expand_dims(basis, 1)
        
        # Data checks
        if basis is not None:
            if basis.shape[0] != covariates.shape[0]:
                raise ValueError("covariates and basis must have the same number of rows")

        pred_dataset = Dataset()
        pred_dataset.add_covariates(covariates)
        pred_dataset.add_basis(basis)
        return self.forest_container.forest_container_cpp.Predict(pred_dataset.dataset_cpp)*self.y_std + self.y_bar
