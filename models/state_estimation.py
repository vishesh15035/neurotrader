"""
Kalman Filter + Particle Filter — State Estimation
Kalman: optimal linear estimator
Particle: non-linear, non-Gaussian Monte Carlo filter
"""
import numpy as np
from scipy.stats import norm

class KalmanFilter:
    """
    Extended Kalman Filter for price + trend estimation
    State: [price, velocity, acceleration]
    Observation: [log_return]
    Full predict-update cycle
    """
    def __init__(self, dt: float = 1.0, process_noise: float = 0.01,
                 obs_noise: float = 0.1):
        self.dt = dt
        # State transition matrix F
        self.F = np.array([
            [1, dt, 0.5*dt**2],
            [0,  1, dt       ],
            [0,  0, 1        ]
        ])
        # Observation matrix H (we observe price)
        self.H  = np.array([[1, 0, 0]])
        # Process noise covariance Q
        q       = process_noise
        self.Q  = q * np.array([
            [dt**5/20, dt**4/8, dt**3/6],
            [dt**4/8,  dt**3/3, dt**2/2],
            [dt**3/6,  dt**2/2, dt     ]
        ])
        # Observation noise covariance R
        self.R  = np.array([[obs_noise**2]])
        # Initial state and covariance
        self.x  = None
        self.P  = np.eye(3) * 1.0
        self.innovations = []
        self.filtered    = []

    def initialize(self, price: float):
        self.x = np.array([price, 0.0, 0.0])
        self.P = np.eye(3) * 1.0

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, z: float):
        z_arr = np.array([z])
        # Innovation
        y  = z_arr - self.H @ self.x
        S  = self.H @ self.P @ self.H.T + self.R
        # Kalman gain
        K  = self.P @ self.H.T @ np.linalg.inv(S)
        # Update
        self.x = self.x + K @ y
        self.P = (np.eye(3) - K @ self.H) @ self.P
        self.innovations.append(float(y[0]))
        self.filtered.append(float(self.x[0]))
        return self.x.copy()

    def fit(self, prices: np.ndarray) -> dict:
        self.initialize(prices[0])
        filtered_prices = []
        velocities      = []
        for p in prices:
            self.predict()
            state = self.update(p)
            filtered_prices.append(state[0])
            velocities.append(state[1])
        self.filtered_prices = np.array(filtered_prices)
        self.velocities      = np.array(velocities)
        # Innovation statistics
        innov = np.array(self.innovations)
        return {
            "filtered_prices": filtered_prices,
            "velocities":      velocities,
            "innov_mean":      round(float(innov.mean()),6),
            "innov_std":       round(float(innov.std()),6),
            "trend":           "UP" if velocities[-1]>0 else "DOWN",
            "trend_strength":  round(abs(float(velocities[-1])),4),
            "acceleration":    round(float(self.x[2]),6),
        }

    def predict_next(self, n: int = 5) -> list:
        forecasts = []
        x, P = self.x.copy(), self.P.copy()
        for _ in range(n):
            x = self.F @ x
            P = self.F @ P @ self.F.T + self.Q
            forecasts.append(round(float(x[0]),2))
        return forecasts


class ParticleFilter:
    """
    Bootstrap Particle Filter — non-linear state estimation
    Handles non-Gaussian noise, regime changes
    N=1000 particles with systematic resampling
    """
    def __init__(self, n_particles: int = 1000, process_std: float = 0.02,
                 obs_std: float = 0.01):
        self.N           = n_particles
        self.process_std = process_std
        self.obs_std     = obs_std
        self.particles   = None
        self.weights     = None
        self.ess_history = []

    def initialize(self, price: float):
        # Particles: [price, velocity]
        self.particles = np.column_stack([
            np.random.normal(price, price*0.02, self.N),
            np.random.normal(0, 0.001, self.N)
        ])
        self.weights = np.ones(self.N) / self.N

    def _transition(self):
        """State transition with noise"""
        self.particles[:,1] += np.random.normal(0, self.process_std*0.1, self.N)
        self.particles[:,0] += self.particles[:,1]
        self.particles[:,0] += np.random.normal(0, self.process_std, self.N)

    def _weight_update(self, obs: float):
        """Likelihood weighting"""
        likelihoods  = norm.pdf(obs, self.particles[:,0], self.obs_std*obs)
        self.weights *= likelihoods
        self.weights += 1e-300  # avoid underflow
        self.weights /= self.weights.sum()

    def _systematic_resample(self):
        """Systematic resampling — O(N) vs O(N log N) naive"""
        N   = self.N
        ess = 1.0 / (self.weights**2).sum()
        self.ess_history.append(ess)
        if ess > N/2: return  # skip if ESS high enough
        cumsum   = np.cumsum(self.weights)
        u        = (np.random.rand() + np.arange(N)) / N
        indices  = np.searchsorted(cumsum, u)
        self.particles = self.particles[indices]
        self.weights   = np.ones(N) / N

    def update(self, obs: float) -> dict:
        self._transition()
        self._weight_update(obs)
        self._systematic_resample()
        mean_price    = float(np.average(self.particles[:,0], weights=self.weights))
        mean_velocity = float(np.average(self.particles[:,1], weights=self.weights))
        std_price     = float(np.sqrt(np.average(
            (self.particles[:,0]-mean_price)**2, weights=self.weights)))
        return {
            "mean_price":    round(mean_price,4),
            "std_price":     round(std_price,4),
            "mean_velocity": round(mean_velocity,6),
            "ess":           round(self.ess_history[-1] if self.ess_history else self.N, 1)
        }

    def fit(self, prices: np.ndarray) -> dict:
        self.initialize(prices[0])
        estimates = []
        for p in prices[1:]:
            est = self.update(float(p))
            estimates.append(est)
        last     = estimates[-1]
        velocity = last["mean_velocity"]
        return {
            "filtered_price": last["mean_price"],
            "uncertainty":    last["std_price"],
            "velocity":       velocity,
            "trend":          "UP" if velocity>0 else "DOWN",
            "avg_ess":        round(np.mean(self.ess_history),1),
            "n_particles":    self.N,
            "signal":         "BUY" if velocity>0.05 else "SELL" if velocity<-0.05 else "HOLD"
        }
