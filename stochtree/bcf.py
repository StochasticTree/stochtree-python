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

class BCFModel:
    def __init__(self) -> None:
        # Internal flag for whether the sample() method has been run
        self.sampled = False
        self.rng = np.random.default_rng()
    
    def is_sampled(self) -> bool:
        return self.sampled
    
    def sample(self, X_train: np.array, Z_train: np.array, y_train: np.array, pi_train: np.array = None, 
               X_test: np.array = None, Z_test: np.array = None, pi_test: np.array = None, feature_types: np.array = None, 
               cutpoint_grid_size = 100, sigma_leaf_mu: float = None, sigma_leaf_tau: float = None, 
               alpha_mu: float = 0.95, alpha_tau: float = 0.25, beta_mu: float = 2.0, beta_tau: float = 3.0, 
               min_samples_leaf_mu: int = 5, min_samples_leaf_tau: int = 5, nu: float = 3, lamb: float = None, 
               a_leaf_mu: float = 3, a_leaf_tau: float = 3, b_leaf_mu: float = None, b_leaf_tau: float = None, 
               q: float = 0.9, sigma2: float = None, num_trees_mu: int = 250, num_trees_tau: int = 50, 
               num_gfr: int = 5, num_burnin: int = 0, num_mcmc: int = 100, sample_sigma_global: bool = True, 
               sample_sigma_leaf_mu: bool = True, sample_sigma_leaf_tau: bool = True, propensity_covariate: str = "mu", 
               adaptive_coding: bool = True, b_0: float = -0.5, b_1: float = 0.5, random_seed: int = -1) -> None:
        # Convert everything to standard shape (2-dimensional)
        if X_train.ndim == 1:
            X_train = np.expand_dims(X_train, 1)
        if Z_train.ndim == 1:
            Z_train = np.expand_dims(Z_train, 1)
        if y_train.ndim == 1:
            y_train = np.expand_dims(y_train, 1)
        if pi_train is not None:
            if pi_train.ndim == 1:
                pi_train = np.expand_dims(pi_train, 1)
        if X_test is not None:
            if X_train.ndim == 1:
                X_train = np.expand_dims(X_train, 1)
        if Z_test is not None:
            if Z_test.ndim == 1:
                Z_test = np.expand_dims(Z_test, 1)
        if pi_test is not None:
            if pi_test.ndim == 1:
                pi_test = np.expand_dims(pi_test, 1)
        
        # Data checks
        if X_test is not None:
            if X_test.shape[1] != X_train.shape[1]:
                raise ValueError("X_train and X_test must have the same number of columns")
        if Z_test is not None:
            if Z_test.shape[1] != Z_train.shape[1]:
                raise ValueError("Z_train and Z_test must have the same number of columns")
        if Z_train.shape[0] != X_train.shape[0]:
            raise ValueError("X_train and Z_train must have the same number of rows")
        if y_train.shape[0] != X_train.shape[0]:
            raise ValueError("X_train and y_train must have the same number of rows")
        if pi_train is not None:
            if pi_train.shape[0] != X_train.shape[0]:
                raise ValueError("X_train and pi_train must have the same number of rows")
        if X_test is not None and Z_test is not None:
            if X_test.shape[0] != Z_test.shape[0]:
                raise ValueError("X_test and Z_test must have the same number of rows")
        if X_test is not None and pi_test is not None:
            if X_test.shape[0] != pi_test.shape[0]:
                raise ValueError("X_test and pi_test must have the same number of rows")

        # Determine whether a test set is provided
        self.has_test = X_test is not None
        
        # Unpack data dimensions
        self.n_train = y_train.shape[0]
        self.n_test = X_test.shape[0] if self.has_test else 0
        self.p_x = X_train.shape[1]

        # Check whether treatment is binary
        self.binary_treatment = np.unique(Z_train).size == 2

        # Adaptive coding will be ignored for continuous / ordered categorical treatments
        self.adaptive_coding = adaptive_coding
        if adaptive_coding and not self.binary_treatment:
            self.adaptive_coding = False

        # Check if user has provided propensities that are needed in the model
        if pi_train is None and propensity_covariate != "none":
            raise ValueError("Must provide a propensity score")
        
        # Set feature type defaults if not provided
        if feature_types is None:
            feature_types = np.zeros(self.p_x)
        
        # Update covariates to include propensities if requested
        if propensity_covariate == "mu":
            feature_types_mu = np.append(feature_types, 0).astype('int')
            feature_types_tau = feature_types.astype('int')
            X_train_mu = np.c_[X_train, pi_train]
            X_train_tau = X_train
            if self.has_test:
                X_test_mu = np.c_[X_test, pi_test]
                X_test_tau = X_test
        elif propensity_covariate == "tau":
            feature_types_tau = np.append(feature_types, 0).astype('int')
            feature_types_mu = feature_types.astype('int')
            X_train_tau = np.c_[X_train, pi_train]
            X_train_mu = X_train
            if self.has_test:
                X_test_tau = np.c_[X_test, pi_test]
                X_test_mu = X_test
        elif propensity_covariate == "both":
            feature_types_tau = np.append(feature_types, 0).astype('int')
            feature_types_mu = np.append(feature_types, 0).astype('int')
            X_train_tau = np.c_[X_train, pi_train]
            X_train_mu = np.c_[X_train, pi_train]
            if self.has_test:
                X_test_tau = np.c_[X_test, pi_test]
                X_test_mu = np.c_[X_test, pi_test]
        elif propensity_covariate == "none":
            feature_types_tau = feature_types.astype('int')
            feature_types_mu = feature_types.astype('int')
            X_train_tau = X_train
            X_train_mu = X_train
            if self.has_test:
                X_test_tau = X_test
                X_test_mu = X_test
        else:
            raise ValueError("propensity_covariate must be one of 'mu', 'tau', 'both', or 'none'")
        
        # Set variable weights for the prognostic and treatment effect forests
        variable_weights_mu = np.repeat(1.0/X_train_mu.shape[1], X_train_mu.shape[1])
        variable_weights_tau = np.repeat(1.0/X_train_tau.shape[1], X_train_tau.shape[1])

        # Scale outcome
        self.y_bar = np.squeeze(np.mean(y_train))
        self.y_std = np.squeeze(np.std(y_train))
        resid_train = (y_train-self.y_bar)/self.y_std

        # Calibrate priors for global sigma^2 and sigma_leaf_mu / sigma_leaf_tau
        if lamb is None:
            reg_basis = X_train
            reg_soln = lstsq(reg_basis, np.squeeze(resid_train))
            sigma2hat = reg_soln[1]
            quantile_cutoff = 0.9
            lamb = (sigma2hat*gamma.ppf(1-quantile_cutoff,nu))/nu
        sigma2 = sigma2hat if sigma2 is None else sigma2
        b_leaf_mu = np.squeeze(np.var(resid_train)) / num_trees_mu if b_leaf_mu is None else b_leaf_mu
        b_leaf_tau = np.squeeze(np.var(resid_train)) / (2*num_trees_tau) if b_leaf_tau is None else b_leaf_tau
        sigma_leaf_mu = np.squeeze(np.var(resid_train)) / num_trees_mu if sigma_leaf_mu is None else sigma_leaf_mu
        sigma_leaf_tau = np.squeeze(np.var(resid_train)) / (2*num_trees_tau) if sigma_leaf_tau is None else sigma_leaf_tau
        current_sigma2 = sigma2
        current_leaf_scale_mu = np.array([[sigma_leaf_mu]])
        current_leaf_scale_tau = np.array([[sigma_leaf_tau]])

        # Container of variance parameter samples
        self.num_gfr = num_gfr
        self.num_burnin = num_burnin
        self.num_mcmc = num_mcmc
        self.num_samples = num_gfr + num_burnin + num_mcmc
        self.sample_sigma_global = sample_sigma_global
        self.sample_sigma_leaf_mu = sample_sigma_leaf_mu
        self.sample_sigma_leaf_tau = sample_sigma_leaf_tau
        if sample_sigma_global:
            self.global_var_samples = np.zeros(self.num_samples)
        if sample_sigma_leaf_mu:
            self.leaf_scale_mu_samples = np.zeros(self.num_samples)
        if sample_sigma_leaf_tau:
            self.leaf_scale_tau_samples = np.zeros(self.num_samples)
        
        # Prepare adaptive coding structure
        if self.adaptive_coding:
            if np.size(b_0) > 1 or np.size(b_1) > 1:
                raise ValueError("b_0 and b_1 must be single numeric values")
            if not (isinstance(b_0, (int, float)) or isinstance(b_1, (int, float))):
                raise ValueError("b_0 and b_1 must be numeric values")
            self.b0_samples = np.zeros(self.num_samples)
            self.b1_samples = np.zeros(self.num_samples)
            current_b_0 = b_0
            current_b_1 = b_1
            tau_basis_train = (1-Z_train)*current_b_0 + Z_train*current_b_1
            if self.has_test:
                tau_basis_test = (1-Z_test)*current_b_0 + Z_test*current_b_1
        else:
            tau_basis_train = Z_train
            if self.has_test:
                tau_basis_test = Z_test

        # Prognostic Forest Dataset (covariates)
        forest_dataset_mu_train = Dataset()
        forest_dataset_mu_train.add_covariates(X_train_mu)
        if self.has_test:
            forest_dataset_mu_test = Dataset()
            forest_dataset_mu_test.add_covariates(X_test_mu)

        # Treatment Forest Dataset (covariates and treatment variable)
        forest_dataset_tau_train = Dataset()
        forest_dataset_tau_train.add_covariates(X_train_tau)
        forest_dataset_tau_train.add_basis(tau_basis_train)
        if self.has_test:
            forest_dataset_tau_test = Dataset()
            forest_dataset_tau_test.add_covariates(X_test_tau)
            forest_dataset_tau_test.add_basis(tau_basis_test)

        # Residual
        residual_train = Residual(resid_train)

        # C++ random number generator
        if random_seed is None: 
            cpp_rng = RNG(-1)
        else:
            cpp_rng = RNG(random_seed)
        
        # Sampling data structures
        forest_sampler_mu = ForestSampler(forest_dataset_mu_train, feature_types_mu, num_trees_mu, self.n_train, alpha_mu, beta_mu, min_samples_leaf_mu)
        forest_sampler_tau = ForestSampler(forest_dataset_tau_train, feature_types_tau, num_trees_tau, self.n_train, alpha_tau, beta_tau, min_samples_leaf_tau)

        # Container of forest samples
        self.forest_container_mu = ForestContainer(num_trees_mu, 1, True)
        self.forest_container_tau = ForestContainer(num_trees_tau, Z_train.shape[1], False)
        
        # Variance samplers
        if self.sample_sigma_global:
            global_var_model = GlobalVarianceModel()
        if self.sample_sigma_leaf_mu:
            leaf_var_model_mu = LeafVarianceModel()
        if self.sample_sigma_leaf_tau:
            leaf_var_model_tau = LeafVarianceModel()

        # Initialize the leaves of each tree in the prognostic forest
        init_mu = np.squeeze(np.mean(resid_train)) / num_trees_mu
        self.forest_container_mu.set_root_leaves(0, init_mu)
        forest_sampler_mu.update_residual(forest_dataset_mu_train, residual_train, self.forest_container_mu, False, 0, True)

        # Initialize the leaves of each tree in the treatment forest
        self.forest_container_tau.set_root_leaves(0, 0.)
        forest_sampler_tau.update_residual(forest_dataset_tau_train, residual_train, self.forest_container_tau, True, 0, True)

        # Run GFR (warm start) if specified
        if self.num_gfr > 0:
            for i in range(self.num_gfr):
                # Sample the prognostic forest
                forest_sampler_mu.sample_one_iteration(
                    self.forest_container_mu, forest_dataset_mu_train, residual_train, cpp_rng, feature_types_mu, 
                    cutpoint_grid_size, current_leaf_scale_mu, variable_weights_mu, current_sigma2, 0, True, True
                )

                # Sample variance parameters (if requested)
                if self.sample_sigma_global:
                    self.global_var_samples[i] = global_var_model.sample_one_iteration(residual_train, cpp_rng, nu, lamb)                    
                    current_sigma2 = self.global_var_samples[i]
                if self.sample_sigma_leaf_mu:
                    self.leaf_scale_mu_samples[i] = leaf_var_model_mu.sample_one_iteration(self.forest_container_mu, cpp_rng, a_leaf_mu, b_leaf_mu, i)
                    current_leaf_scale_mu[0,0] = self.leaf_scale_mu_samples[i]
                
                # Sample the treatment forest
                forest_sampler_tau.sample_one_iteration(
                    self.forest_container_tau, forest_dataset_tau_train, residual_train, cpp_rng, feature_types_tau, 
                    cutpoint_grid_size, current_leaf_scale_tau, variable_weights_tau, current_sigma2, 0, True, True
                )
                
                # Sample variance parameters (if requested)
                if self.sample_sigma_global:
                    self.global_var_samples[i] = global_var_model.sample_one_iteration(residual_train, cpp_rng, nu, lamb)                    
                    current_sigma2 = self.global_var_samples[i]
                if self.sample_sigma_leaf_tau:
                    self.leaf_scale_tau_samples[i] = leaf_var_model_tau.sample_one_iteration(self.forest_container_tau, cpp_rng, a_leaf_tau, b_leaf_tau, i)
                    current_leaf_scale_tau[0,0] = self.leaf_scale_tau_samples[i]                
                
                # Sample coding parameters (if requested)
                if self.adaptive_coding:
                    mu_x = self.forest_container_mu.predict_raw_single_forest(forest_dataset_mu_train, i)
                    tau_x = np.squeeze(self.forest_container_tau.predict_raw_single_forest(forest_dataset_tau_train, i))
                    s_tt0 = np.sum(tau_x*tau_x*(Z_train==0))
                    s_tt1 = np.sum(tau_x*tau_x*(Z_train==1))
                    partial_resid_mu = resid_train - np.squeeze(mu_x)
                    s_ty0 = np.sum(tau_x*partial_resid_mu*(Z_train==0))
                    s_ty1 = np.sum(tau_x*partial_resid_mu*(Z_train==1))
                    current_b_0 = self.rng.normal(loc = (s_ty0/(s_tt0 + 2*self.global_var_samples[i])), 
                                             scale = np.sqrt(self.global_var_samples[i]/(s_tt0 + 2*self.global_var_samples[i])), size = 1)
                    current_b_1 = self.rng.normal(loc = (s_ty1/(s_tt1 + 2*self.global_var_samples[i])), 
                                             scale = np.sqrt(self.global_var_samples[i]/(s_tt1 + 2*self.global_var_samples[i])), size = 1)
                    tau_basis_train = (1-Z_train)*current_b_0 + Z_train*current_b_1
                    forest_dataset_tau_train.update_basis(tau_basis_train)
                    if self.has_test:
                        tau_basis_test = (1-Z_test)*current_b_0 + Z_test*current_b_1
                        forest_dataset_tau_test.update_basis(tau_basis_test)
                    self.b0_samples[i] = current_b_0
                    self.b1_samples[i] = current_b_1
        
        # Run MCMC
        if self.num_burnin + self.num_mcmc > 0:
            for i in range(self.num_gfr, self.num_samples):
                # Sample the prognostic forest
                forest_sampler_mu.sample_one_iteration(
                    self.forest_container_mu, forest_dataset_mu_train, residual_train, cpp_rng, feature_types_mu, 
                    cutpoint_grid_size, current_leaf_scale_mu, variable_weights_mu, current_sigma2, 0, False, True
                )

                # Sample variance parameters (if requested)
                if self.sample_sigma_global:
                    self.global_var_samples[i] = global_var_model.sample_one_iteration(residual_train, cpp_rng, nu, lamb)                    
                    current_sigma2 = self.global_var_samples[i]
                if self.sample_sigma_leaf_mu:
                    self.leaf_scale_mu_samples[i] = leaf_var_model_mu.sample_one_iteration(self.forest_container_mu, cpp_rng, a_leaf_mu, b_leaf_mu, i)
                    current_leaf_scale_mu[0,0] = self.leaf_scale_mu_samples[i]
                
                # Sample the treatment forest
                forest_sampler_tau.sample_one_iteration(
                    self.forest_container_tau, forest_dataset_tau_train, residual_train, cpp_rng, feature_types_tau, 
                    cutpoint_grid_size, current_leaf_scale_tau, variable_weights_tau, current_sigma2, 0, False, True
                )
                
                # Sample variance parameters (if requested)
                if self.sample_sigma_global:
                    self.global_var_samples[i] = global_var_model.sample_one_iteration(residual_train, cpp_rng, nu, lamb)                    
                    current_sigma2 = self.global_var_samples[i]
                if self.sample_sigma_leaf_tau:
                    self.leaf_scale_tau_samples[i] = leaf_var_model_tau.sample_one_iteration(self.forest_container_tau, cpp_rng, a_leaf_tau, b_leaf_tau, i)
                    current_leaf_scale_tau[0,0] = self.leaf_scale_tau_samples[i]                
                
                # Sample coding parameters (if requested)
                if self.adaptive_coding:
                    mu_x = self.forest_container_mu.predict_raw_single_forest(forest_dataset_mu_train, i)
                    tau_x = np.squeeze(self.forest_container_tau.predict_raw_single_forest(forest_dataset_tau_train, i))
                    s_tt0 = np.sum(tau_x*tau_x*(Z_train==0))
                    s_tt1 = np.sum(tau_x*tau_x*(Z_train==1))
                    partial_resid_mu = resid_train - np.squeeze(mu_x)
                    s_ty0 = np.sum(tau_x*partial_resid_mu*(Z_train==0))
                    s_ty1 = np.sum(tau_x*partial_resid_mu*(Z_train==1))
                    current_b_0 = self.rng.normal(loc = (s_ty0/(s_tt0 + 2*self.global_var_samples[i])), 
                                             scale = np.sqrt(self.global_var_samples[i]/(s_tt0 + 2*self.global_var_samples[i])), size = 1)
                    current_b_1 = self.rng.normal(loc = (s_ty1/(s_tt1 + 2*self.global_var_samples[i])), 
                                             scale = np.sqrt(self.global_var_samples[i]/(s_tt1 + 2*self.global_var_samples[i])), size = 1)
                    tau_basis_train = (1-Z_train)*current_b_0 + Z_train*current_b_1
                    forest_dataset_tau_train.update_basis(tau_basis_train)
                    if self.has_test:
                        tau_basis_test = (1-Z_test)*current_b_0 + Z_test*current_b_1
                        forest_dataset_tau_test.update_basis(tau_basis_test)
                    self.b0_samples[i] = current_b_0
                    self.b1_samples[i] = current_b_1
        
        # Mark the model as sampled
        self.sampled = True
        
        # Store predictions
        self.mu_hat_train = self.forest_container_mu.forest_container_cpp.Predict(forest_dataset_mu_train.dataset_cpp)*self.y_std + self.y_bar
        tau_raw = self.forest_container_tau.forest_container_cpp.PredictRaw(forest_dataset_tau_train.dataset_cpp)
        self.tau_hat_train = tau_raw*self.y_std
        if self.adaptive_coding:
            self.tau_hat_train = self.tau_hat_train*np.expand_dims(self.b1_samples - self.b0_samples, axis=(0,2))
        self.y_hat_train = self.mu_hat_train + Z_train*np.squeeze(self.tau_hat_train)
        if self.has_test:
            self.mu_hat_test = self.forest_container_mu.forest_container_cpp.Predict(forest_dataset_mu_test.dataset_cpp)*self.y_std + self.y_bar
            tau_raw = self.forest_container_tau.forest_container_cpp.PredictRaw(forest_dataset_tau_test.dataset_cpp)
            self.tau_hat_test = tau_raw*self.y_std
            if self.adaptive_coding:
                self.tau_hat_test = self.tau_hat_test*np.expand_dims(self.b1_samples - self.b0_samples, axis=(0,2))
            self.y_hat_test = self.mu_hat_test + Z_test*np.squeeze(self.tau_hat_test)
    
    def predict_mu(self, X: np.array) -> np.array:
        if not self.is_sampled():
            msg = (
                "This BCFModel instance is not fitted yet. Call 'fit' with "
                "appropriate arguments before using this model."
            )
            raise NotSampledError(msg)
        dataset = Dataset()
        dataset.add_covariates(X)
        return self.forest_container_mu.forest_container_cpp.Predict(dataset.dataset_cpp)*self.y_std + self.y_bar
    
    def predict_tau(self, X: np.array, Z: np.array) -> np.array:
        dataset = Dataset()
        dataset.add_covariates(X)
        dataset.add_basis(Z)
        tau_raw = self.forest_container_tau.forest_container_cpp.PredictRaw(dataset.dataset_cpp)
        tau_x = tau_raw*self.y_std
        if self.adaptive_coding:
            tau_x = tau_x*np.expand_dims(self.b1_samples - self.b0_samples, axis=(0,2))
        return tau_x
    
    def predict(self, X: np.array, Z: np.array, propensity: np.array) -> np.array:
        mu_dataset = Dataset()
        Xtilde = np.c_[X, propensity]
        mu_dataset.add_covariates(Xtilde)
        mu_dataset.add_basis(Z)
        tau_dataset = Dataset()
        tau_dataset.add_covariates(X)
        tau_dataset.add_basis(Z)
        mu_x = self.forest_container_mu.forest_container_cpp.Predict(mu_dataset.dataset_cpp)*self.y_std + self.y_bar
        tau_raw = self.forest_container_tau.forest_container_cpp.PredictRaw(tau_dataset.dataset_cpp)
        tau_x = tau_raw*self.y_std
        if self.adaptive_coding:
            tau_x = tau_x*np.expand_dims(self.b1_samples - self.b0_samples, axis=(0,2))
        return mu_x + Z*tau_x
