import time
import numpy as np
from scipy import stats as st
from typing import Union, Optional, List, Sequence


class InterpolatedKDE(st.gaussian_kde):
    """
    A custom Kernel Density Estimation class that uses interpolation for 
    efficient density estimation, inheriting from scipy's gaussian_kde.
    """
    
    def __init__(self, dataset, bw_method=None, kernel_width_scalar=1.0, 
                 precomputed_points=None):
        """
        Initialize the interpolated KDE.
        
        Args:
            dataset: Training data for bandwidth estimation, or None if using precomputed points
            bw_method: Bandwidth method (defaults to 'scott')
            kernel_width_scalar: Multiplier for bandwidth
            precomputed_points: Optional tuple of (x_points, y_points) for direct initialization
        """
        if precomputed_points is not None:
            # Initialize with minimal valid dataset just to set up the class
            # Use two points that won't affect our interpolation
            dummy_data = np.array([-1e9, 1e9])
            super().__init__(dummy_data, bw_method=bw_method)
            self.x_points = np.array(precomputed_points[0])
            self.y_points = np.array(precomputed_points[1])
            
            # Set the covariance factor to a tiny value since we're not using it
            self.covariance_factor = lambda: 1e-10
            self._compute_covariance()
        else:
            # Use parent class initialization 
            super().__init__(dataset, bw_method=bw_method)
            
            # Adjust bandwidth if needed
            if kernel_width_scalar != 1.0:
                self.set_bandwidth(self.factor * kernel_width_scalar)
            
            # Create a dense set of points for pre-computing KDE
            x_range_min = np.min(dataset)
            x_range_max = np.max(dataset)
            
            # Create 1000 points spanning the data range
            self.x_points = np.linspace(x_range_min, x_range_max, 1000)
            
            # Compute KDE on these points using the parent class method
            self.y_points = super().evaluate(self.x_points)
            
            # Ensure x_points are sorted for interpolation
            sort_idx = np.argsort(self.x_points)
            self.x_points = self.x_points[sort_idx]
            self.y_points = self.y_points[sort_idx]
    
    def evaluate(self, points):
        """
        Evaluate densities using interpolation.
        
        Args:
            points: Points to evaluate density for
        
        Returns:
            Interpolated density values
        """
        # Ensure points is a numpy array
        points = np.asarray(points)
        
        # Use numpy's interpolation
        # Fill values outside the known range with the nearest edge value
        return np.interp(points, self.x_points, self.y_points,
                        left=self.y_points[0], right=self.y_points[-1])



class ActivationDistribution:
    def __init__(self, 
                 activation_values: Optional[Union[np.ndarray, list]] = None,
                 kde: Optional[Union[st.gaussian_kde, InterpolatedKDE]] = None, 
                 bw_method: Optional[Union[str, float]] = 'scott',
                 kernel_width_scalar: float = 1.0,
                 mean: Optional[float] = None,
                 std: Optional[float] = None):
        """
        Initialize distribution from activation values.
        
        New parameters:
            mean: Pre-computed mean of training data
            std: Pre-computed standard deviation of training data
        """
        self.mean = mean
        self.std = std
        
        if kde is not None and activation_values is not None:
            raise ValueError("Specify either activation_values or kde, not both")
            
        if kde is not None:
            if isinstance(kde, InterpolatedKDE):
                self.kde = kde
            elif isinstance(kde, st.gaussian_kde):
                self.kde = InterpolatedKDE(
                    kde.dataset[0], 
                    bw_method=kde.factor, 
                    kernel_width_scalar=kernel_width_scalar
                )
            else:
                raise TypeError("kde must be either scipy.stats.gaussian_kde or InterpolatedKDE")
            return
            
        if not isinstance(activation_values, (np.ndarray, list)):
            raise TypeError("activation_values must be numpy array or list")
            
        values = np.asarray(activation_values)
        
        if values.size == 0:
            raise ValueError("activation_values cannot be empty")
            
        if not np.all(np.isfinite(values)):
            raise ValueError("activation_values contains non-finite values (inf or nan)")
            
        self.values = values.flatten()
        
        # Compute statistics if not provided
        if self.mean is None:
            self.mean = np.mean(self.values)
        if self.std is None:
            self.std = np.std(self.values)
        
        try:
            self._set_kde(bw_method, kernel_width_scalar)
        except Exception as e:
            raise ValueError(f"Failed to create KDE: {str(e)}")

    def log_likelihood(self,
                      new_activations: Union[np.ndarray, list],
                      epsilon: float = 1e-6,
                      std_threshold: Optional[float] = None,
                      detect_tails: bool = False) -> np.ndarray:
        """
        Compute log likelihood for new activation values.
        
        New parameter:
            std_threshold: If set, only consider points beyond this many standard 
                         deviations from the mean
        """
        if not isinstance(new_activations, (np.ndarray, list)):
            raise TypeError("new_activations must be numpy array or list")
            
        values = np.asarray(new_activations)
        
        if values.size == 0:
            raise ValueError("new_activations cannot be empty")
            
        if not np.all(np.isfinite(values)):
            raise ValueError("new_activations contains non-finite values")
            
        flat_activations = values.flatten()
        if std_threshold is not None:
            # Create mask for points in the tails
            lower_bound = self.mean - std_threshold * self.std
            upper_bound = self.mean + std_threshold * self.std
            tail_mask = (flat_activations < lower_bound) | (flat_activations > upper_bound)
            # If no points in tails, return very low log likelihood
            # if not np.any(tail_mask):
            #    return np.full_like(flat_activations, np.log(epsilon))
            
            # Returning very low LL if only a few points. Right?
            if np.sum(tail_mask) < 3:
                return np.full_like(flat_activations, np.log(epsilon))
            
            # Calculate mean log likelihood of points in tails
            tail_points = flat_activations[tail_mask]
            tail_lls = np.log(self.kde.evaluate(tail_points) + epsilon)
            mean_tail_ll = np.mean(tail_lls)
            
            # Return mean tail likelihood for all points
            return np.full_like(flat_activations, mean_tail_ll)
        if detect_tails:
            # Create mask for points in the tails
            lower_bound = self.left_tail_cutoff
            upper_bound = self.right_tail_cutoff
            tail_mask = (flat_activations <= lower_bound) | (flat_activations >= upper_bound)
            # If no points in tails, return very low log likelihood
            # if not np.any(tail_mask):
            #    return np.full_like(flat_activations, np.log(epsilon))
            
            # Returning very low LL if only a few points. Right?
            if np.sum(tail_mask) < 3:
                return np.full_like(flat_activations, np.log(epsilon))
            
            # Calculate mean log likelihood of points in tails

            tail_points = flat_activations[tail_mask]
            tail_lls = np.log(self.kde.evaluate(tail_points) + epsilon)
            mean_tail_ll = np.mean(tail_lls)
            
            # Return mean tail likelihood for all points
            return np.full_like(flat_activations, mean_tail_ll)

        
        # Normal case - evaluate all points
        return np.log(self.kde.evaluate(flat_activations) + epsilon)

    
    def set_tail_cutoffs(self, density_threshold: float = 1e-4) -> None:
        """
        Detect cutoff points for left and right tails where minor modes begin.
        Starts from zero and moves outward until density drops below threshold,
        then continues until 3 of next 4 points show increasing density.
        
        Args:
            density_threshold: Threshold below which we consider to be in the tail
        """
        # Find index closest to zero
        zero_idx = np.abs(self.kde.x_points).argmin()
        densities = self.kde.y_points
        
        # Function to check if we've found a minor mode
        def has_minor_mode(points: np.ndarray) -> bool:
            if len(points) < 4:
                return False
            increases = points[1:] > points[0]
            return np.sum(increases[:4]) >= 3
        
        # Find left tail cutoff (moving towards negative values)
        left_idx = zero_idx
        while left_idx > 0:
            if densities[left_idx] < density_threshold:
                # Now check for minor mode
                remaining_points = densities[max(0, left_idx-4):left_idx]
                if has_minor_mode(remaining_points[::-1]):  # Reverse array since we're going left
                    break
            left_idx -= 1
        self.left_tail_cutoff = self.kde.x_points[left_idx]
        
        # Find right tail cutoff (moving towards positive values)
        right_idx = zero_idx
        while right_idx < len(self.kde.x_points) - 1:
            if densities[right_idx] < density_threshold:
                # Check for minor mode
                remaining_points = densities[right_idx:min(len(densities), right_idx+5)]
                if has_minor_mode(remaining_points):
                    break
            right_idx += 1
        self.right_tail_cutoff = self.kde.x_points[right_idx]    


    def _set_kde(self, bw_method, kernel_width_scalar):
        """
        Internal method to set the kernel density estimate. Currently
        just set via input arguments. Separated to enable easier use
        of estimated or cross-validated method of setting the params
        """
        bw_method = 'scott' if bw_method is None else bw_method
        self.kde = InterpolatedKDE(
            self.values, 
            bw_method=bw_method, 
            kernel_width_scalar=kernel_width_scalar
        )

    
    def total_log_likelihood(self, new_activations: Union[np.ndarray, list]) -> float:
        """
        Compute total log likelihood (sum of individual log likelihoods).
        
        Args:
            new_activations: New activation values to evaluate
                
        Returns:
            Total log likelihood value
            
        Raises:
            Same as log_likelihood method
        """
        return np.sum(self.log_likelihood(new_activations))

    
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """
        Generate random samples from the distribution.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Array of samples
            
        Raises:
            ValueError: If n_samples is not positive
        """
        if not isinstance(n_samples, int):
            raise TypeError("n_samples must be an integer")
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
            
        return self.kde.resample(n_samples).flatten()




class LayerwiseDistributions:
    """
    Wrapper class to handle activation distributions for each layer of a model.
    Provides indexed access to layer-specific ActivationDistribution objects.
    """
    
    def __init__(self, 
                 data: Optional[Union[List[np.ndarray], np.ndarray, 
                                      List[st.gaussian_kde], 
                                      List[ActivationDistribution]]] = None,
                 n_layers: Optional[int] = None,
                 bw_method: Optional[Union[str, float]] = None,
                 kernel_width_scalar: float = 1.0,
                 copy_data: bool = True):
        """
        Initialize distributions for all layers.
        
        Args:
            data: Optional initialization data. Can be:
                - List of 1D arrays (one per layer)
                - 2D array (layers x values)
                - List of pre-computed KDEs
                - List of ActivationDistribution objects
            n_layers: Number of layers (required if data is None)
            bw_method: Bandwidth parameter for KDE estimation
            kernel_width_scalar: Multiplier for the automatic bandwidth
            copy_data: Deep copy data passed in?
            
        Raises:
            ValueError: If neither data nor n_layers is provided
        """
        if data is None and n_layers is None:
            raise ValueError("Must provide either data or n_layers")
            
        self.distributions = []
        
        if data is None:
            # Initialize empty distributions
            self.distributions = [None] * n_layers

        else:
            self._initialize_from_data(data, bw_method, kernel_width_scalar, copy_data)
            
    def _initialize_from_data(self, data, bw_method, kernel_width_scalar, copy_data=True):
        """Helper method to initialize from different data types."""
        
        if isinstance(data, list) and all(isinstance(x, ActivationDistribution) for x in data):
            # List of ActivationDistribution objects
            #print("About to call AD constructor with AD objects")
            self.distributions = data.copy() if copy_data else data
            
        elif isinstance(data, list) and all(isinstance(x, st.gaussian_kde) for x in data):
            # List of KDEs
            #print("About to call AD constructor with KDEs")
            self.distributions = [
                ActivationDistribution(kde=kde, kernel_width_scalar=kernel_width_scalar) 
                for kde in data
            ]
            
        elif isinstance(data, (list, np.ndarray)):
            # Raw data
            #print("About to call AD constructor with ndarray")
            if isinstance(data, np.ndarray) and data.ndim != 2:
                raise ValueError("If providing numpy array, must be 2D (layers x values)")
                
            data_list = data if isinstance(data, list) else [data[i] for i in range(data.shape[0])]
            self.distributions = [
                ActivationDistribution(
                    activation_values=layer_data, 
                    bw_method=bw_method,
                    kernel_width_scalar=kernel_width_scalar
                )
                for layer_data in data_list
            ]
            
        else:
            raise ValueError("Unrecognized data format for initialization")
    
    def __getitem__(self, layer_idx: int) -> ActivationDistribution:
        """Get distribution for specific layer."""
        if not isinstance(layer_idx, int):
            raise TypeError("Layer index must be integer")
            
        if layer_idx < 0 or layer_idx >= len(self.distributions):
            raise IndexError(f"Layer index {layer_idx} out of range")
            
        if self.distributions[layer_idx] is None:
            raise ValueError(f"No distribution set for layer {layer_idx}")
            
        return self.distributions[layer_idx]
    
    def __len__(self) -> int:
        """Number of layers."""
        return len(self.distributions)
    
    def set_distribution(self, 
                        layer_idx: int, 
                        data: Union[np.ndarray, st.gaussian_kde, ActivationDistribution],
                        bw_method: Optional[Union[str, float]] = None,
                        kernel_width_scalar: float = 1.0):
        """
        Set distribution for a specific layer.
        
        Args:
            layer_idx: Layer index
            data: Either raw activation values, pre-computed KDE, or ActivationDistribution
            bw_method: Bandwidth parameter/method for st.gaussian_kde (only used for raw data)
            kernel_width_scalar: Multiplier for the automatic bandwidth
        """
        if not isinstance(layer_idx, int):
            raise TypeError("Layer index must be integer")
            
        if layer_idx < 0 or layer_idx >= len(self.distributions):
            raise IndexError(f"Layer index {layer_idx} out of range")
            
        if isinstance(data, ActivationDistribution):
            self.distributions[layer_idx] = data
        elif isinstance(data, st.gaussian_kde):
            self.distributions[layer_idx] = ActivationDistribution(
                kde=data, 
                kernel_width_scalar=kernel_width_scalar
            )
        else:
            self.distributions[layer_idx] = ActivationDistribution(
                data, 
                bw_method=bw_method,
                kernel_width_scalar=kernel_width_scalar
            )
    
    def get_all_log_likelihoods(self, 
                               new_activations: Sequence[np.ndarray],
                               sum_lls : bool=True) -> List[float]:
        """
        Compute total log likelihood for each layer.
        
        Args:
            new_activations: List of activation arrays, one per layer
            sum_lls: return the sum of the LLs of the layers? Returns a list o/w
            
        Returns:
            List of total log likelihoods, one per layer, or their sum
        """
        if len(new_activations) != len(self.distributions):
            raise ValueError("Number of activation arrays must match number of layers")

        lls = [self[i].total_log_likelihood(acts) for i, acts in enumerate(new_activations)]
        return np.sum(lls) if sum_lls else lls


