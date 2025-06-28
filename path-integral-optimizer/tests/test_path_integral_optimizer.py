import numpy as np
import pytest
from path_integral_optimizer.path_integral_optimizer import PathIntegralOptimizer
from path_integral_optimizer.path_integral_optimizer_result import PathIntegralOptimizerResult
from path_integral_optimizer.dataset.synthetic_dataset import SyntheticDataset

class TestPathIntegralOptimizer:
    """Test suite for PathIntegralOptimizer class"""
    
    @pytest.fixture
    def sample_config(self):
        """Provides a standard configuration for PathIntegralOptimizer"""
        return {
            "base_cost_prior": {"dist": "TruncatedNormal", "mu": 0.5, "sigma": 0.05, "lower": 0.0},
            "base_benefit_prior": {"dist": "TruncatedNormal", "mu": 1.0, "sigma": 0.1, "lower": 0.0},
            "scale_benefit_prior": {"dist": "Beta", "mu": 0.5, "sigma": 0.1},
            "gp_eta_prior": {"dist": "HalfNormal", "sigma": 1},
            "gp_ell_prior": {"dist": "Gamma", "mu": 2, "sigma": 0.5},
            "gp_mean_prior": {"dist": "Normal", "mu": 1, "sigma": 0.5},
            "total_resource": 100.0,
            "T": 12,
            "hbar": 0.5,
            "num_steps": 500,
            "burn_in": 250
        }
    
    @pytest.fixture
    def synthetic_data(self):
        """Generates synthetic data for testing"""
        dataset = SyntheticDataset(T=12, total_resource=100.0)
        return dataset.generate()
    
    def test_initialization(self, sample_config):
        """Test basic initialization of PathIntegralOptimizer"""
        optimizer = PathIntegralOptimizer(**sample_config)
        assert optimizer is not None
    
    def test_invalid_T(self):
        """Test initialization with invalid T values"""
        config = self.sample_config()
        config["T"] = 0  # Change to invalid value
        
        with pytest.raises(ValueError):
            PathIntegralOptimizer(**config)
    
    def test_run_mcmc(self, sample_config):
        """Test MCMC sampling runs and produces results"""
        optimizer = PathIntegralOptimizer(**sample_config)
        optimizer.run_mcmc()
        
        assert optimizer.trace is not None
        assert isinstance(optimizer.trace, az.InferenceData)
        assert optimizer.mcmc_paths is not None
        assert optimizer.actions is not None
    
    def test_from_data(self, synthetic_data, sample_config):
        """Test parameter estimation from synthetic data"""
        data = synthetic_data
        
        # Test creation from data
        pio = PathIntegralOptimizer.from_data(
            input=data.input,
            cost=data.cost,
            benefit=data.benefit,
            total_resource=data.total_resource,
            hbar=sample_config["hbar"],
            num_steps=sample_config["num_steps"],
            burn_in=sample_config["burn_in"],
            forecast_steps=sample_config["T"],
            t=data.t  # Use the time points from the synthetic data
        )
        
        assert pio is not None
        assert pio.trace is not None
        assert pio.mcmc_paths is not None
    
    def test_plotting(self, sample_config):
        """Test that plotting methods run without error"""
        optimizer = PathIntegralOptimizer(**sample_config)
        optimizer.run_mcmc()
        
        # Test basic plotting
        try:
            optimizer.plot()
        except Exception as e:
            pytest.fail(f"plot() raised {e} unexpectedly!")
            
        # Test top paths plotting
        try:
            optimizer.plot_top_paths()
        except Exception as e:
            pytest.fail(f"plot_top_paths() raised {e} unexpectedly!")
            
        # Test forecast plotting
        try:
            optimizer.plot_forecast()
        except Exception as e:
            pytest.fail(f"plot_forecast() raised {e} unexpectedly!")
    
    def test_summary_generation(self, sample_config):
        """Test summary generation and result object"""
        optimizer = PathIntegralOptimizer(**sample_config)
        optimizer.run_mcmc()
        
        summary = optimizer.generate_summary()
        assert isinstance(summary, PathIntegralOptimizerResult)
        
        # Test that the summary contains reasonable values
        assert np.isfinite(summary.mean_path).all()
        assert np.isfinite(summary.std_path).all()
        assert summary.num_samples > 0
        assert summary.T == sample_config["T"]
    
    def test_invalid_input_handling(self, sample_config):
        """Test that invalid inputs are handled gracefully"""
        # Test with mismatched historical data
        config = sample_config.copy()
        config["historical_t"] = np.array([1, 2, 3])
        config["historical_input"] = np.array([10, 20])  # Different length
        
        with pytest.raises(ValueError):
            PathIntegralOptimizer(**config)
    
    def test_forecast_without_mcmc(self, sample_config):
        """Test that plotting forecast without running MCMC fails gracefully"""
        optimizer = PathIntegralOptimizer(**sample_config)
        
        with pytest.raises(Exception) as exc_info:
            optimizer.plot_forecast()
        
        assert "Run run_mcmc() first" in str(exc_info.value)
    
    def test_invalid_action_computation(self, sample_config):
        """Test handling of invalid action computation"""
        optimizer = PathIntegralOptimizer(**sample_config)
        
        # Test with non-finite parameters
        with pytest.raises(RuntimeError):
            optimizer._compute_action_numpy(
                x_path=np.array([1, 2, 3]),
                base_cost=np.inf,
                base_benefit=1.0,
                scale_benefit=0.5,
                d_t=np.array([1.2, 1.2, 1.2])
            )
    
    def test_invalid_forecast_handling(self, sample_config):
        """Test plotting forecast with edge cases"""
        optimizer = PathIntegralOptimizer(**sample_config)
        optimizer.run_mcmc()
        
        # Test with invalid T
        optimizer.T = 0
        with pytest.raises(ValueError):
            optimizer.plot_forecast()
        
        # Test with invalid trace
        optimizer.trace = None
        with pytest.raises(ValueError):
            optimizer.plot_forecast()
    
    def test_historical_forecasted_metrics(self, sample_config):
        """Test historical and forecasted metrics calculation"""
        optimizer = PathIntegralOptimizer(**sample_config)
        optimizer.run_mcmc()
        
        historical_cost, historical_benefit, forecast_cost, forecast_benefit = optimizer._calculate_historical_forecasted_metrics()
        
        assert historical_cost is None or isinstance(historical_cost, np.ndarray)
        assert historical_benefit is None or isinstance(historical_benefit, np.ndarray)
        assert isinstance(forecast_cost, np.ndarray)
        assert isinstance(forecast_benefit, np.ndarray)
        
        # If historical data is provided, test those metrics too
        if optimizer.historical_input is not None and optimizer.historical_t is not None:
            assert historical_cost.shape == optimizer.historical_input.shape
            assert historical_benefit.shape == optimizer.historical_input.shape
            assert forecast_cost.shape == (optimizer.T,)
            assert forecast_benefit.shape == (optimizer.T,)
    
    def test_invalid_historical_data(self, sample_config):
        """Test handling of invalid historical data"""
        # Test with mismatched historical data
        config = sample_config.copy()
        config["historical_t"] = np.array([1, 2, 3])
        config["historical_input"] = np.array([10, 20])  # Different length
        
        with pytest.raises(ValueError):
            PathIntegralOptimizer(**config)
    
    def test_forecast_plotting(self, sample_config):
        """Test forecast plotting with historical data"""
        optimizer = PathIntegralOptimizer(**sample_config)
        optimizer.run_mcmc()
        
        # Test basic forecast plotting
        try:
            optimizer.plot_forecast()
        except Exception as e:
            pytest.fail(f"plot_forecast() raised {e} unexpectedly!")
            
        # Test saving to file
        try:
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(suffix='.png') as tmpfile:
                optimizer.plot_forecast(output_file=tmpfile.name)
                assert os.path.exists(tmpfile.name)
        except Exception as e:
            pytest.fail(f"plot_forecast(output_file=...) raised {e} unexpectedly!")
