import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

class CVFilter:
    def __init__(self):
        # Initialize filter parameters
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.pf = np.eye(6)  # Filter state covariance matrix
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(3, 6)  # Measurement matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time

    def Initialize_Filter_state_covariance(self, x, y, z, vx, vy, vz, time):
        # Initialize filter state
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Filtered_Time = time

    def Filter_state_covariance(self, measurements):
        # JPDA algorithm implementation
        for measurement in measurements:
            M_rng1, M_az, M_el, M_time = measurement

            # Predict step
            dt = M_time - self.Meas_Time
            Phi = np.eye(6) + np.eye(6) * dt
            Q = np.eye(6) * self.plant_noise
            self.Sf = np.dot(Phi, self.Sf)
            self.pf = np.dot(np.dot(Phi, self.pf), Phi.T) + Q

            # Update step
            Z = np.array([[M_rng1], [M_az], [M_el]])
            Inn = Z - np.dot(self.H, self.Sf)
            S = np.dot(self.H, np.dot(self.pf, self.H.T)) + self.R
            K = np.dot(np.dot(self.pf, self.H.T), np.linalg.inv(S))
            self.Sf = self.Sf + np.dot(K, Inn)
            self.pf = np.dot(np.eye(6) - np.dot(K, self.H), self.pf)

        # Compute probabilities for each measurement being associated with each target
        conditional_probs = []
        for measurement in measurements:
            M_rng1, M_az, M_el, M_time = measurement
            Z = np.array([[M_rng1], [M_az], [M_el]])
            H = np.eye(3, 6)
            Inn = Z - np.dot(H, self.Sf)
            S = np.dot(H, np.dot(self.pf, H.T)) + self.R
            prob = multivariate_normal.pdf(Inn.flatten(), mean=None, cov=S)
            conditional_probs.append(prob)

        # Calculate marginal probabilities
        marginal_probs = np.prod(conditional_probs, axis=0)

        # Find the most likely association
        most_likely_index = np.argmax(marginal_probs)
        most_likely_measurement = measurements[most_likely_index]

        # Print states
        print("Filter state:", self.Sf.flatten())
        print("Filter state covariance:", self.pf)

        return self.Sf, most_likely_measurement


# Create an instance of the CVFilter class
kalman_filter = CVFilter()

# Define initial state estimates
x = 0  # Initial x position
y = 0  # Initial y position
z = 0  # Initial z position
vx = 0  # Initial velocity in x direction
vy = 0  # Initial velocity in y direction
vz = 0  # Initial velocity in z direction
initial_time = 0  # Initial time

# Initialize the filter with initial state estimates
kalman_filter.Initialize_Filter_state_covariance(x, y, z, vx, vy, vz, initial_time)

# Define measurements (M_rng1, M_az, M_el, M_time) for each time step
measurements = [
    (26.90994, 304.7405, 0.02466, 54272.85),
    (26.90902, 304.6282, 0.1, 54273.35),
    (26.907703, 304.6981, 0.07435, 54274.05),
    (26.90806, 304.6647, 0.032784, 54274.52),
    (26.90842, 304.6671, 0.01765, 54275.05),
    (26.91023, 304.5495, 0.1, 54275.53),
    (26.89167, 304.6397, 0.1, 54276.05),
    (26.90909, 304.4994, 0.010024, 54277.54),
    (34.49111, 303.6271, 0.083879, 54366.75),
    (34.44652, 303.6783, 0.768014, 54367.26),
    (17.00434, 310.531, 0.01, 54405.75),
    (17.60408, 310.4814, 0.1, 54406.24),
    (17.00446, 310.3168, 0.01, 54406.74)
]

# Process measurements and get predicted state estimates at each time step
for measurement in measurements:
    filtered_state, most_likely_measurement = kalman_filter.Filter_state_covariance([measurement])
    
    # Print measured and most likely measurement
    print("Measured:", measurement)
    print("Most likely measurement:", most_likely_measurement)
