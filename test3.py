import numpy as np

class CVFilter:
    def __init__(self):
        self.Sf = np.zeros((6, 1))  # Filter state vector
        self.pf = np.eye(6)  # Filter state covariance matrix
        self.plant_noise = 20  # Plant noise covariance
        self.H = np.eye(6)  # State of the filter matrix
        self.R = np.eye(3)  # Measurement noise covariance
        self.Meas_Time = 0  # Measured time

    def Initialize_Filter_state_covariance(self, x, y, z, vx, vy, vz, time):
        # Initialize the filter state with the provided values
        self.Sf = np.array([[x], [y], [z], [vx], [vy], [vz]])
        self.Filtered_Time = time

    def Filter_state_covariance(self, M_rng1, M_az, M_el, M_time):
        # Update the filter with the provided measurements
        dt = M_time - self.Meas_Time  # Time difference since the last measurement
        self.Meas_Time = M_time
        
        # State transition matrix (assuming constant velocity model)
        Phi = np.eye(6) + np.eye(6) * dt
        
        # Predict step
        Q = np.eye(6) * self.plant_noise  # Process noise covariance
        self.Sf = np.dot(Phi, self.Sf)  # Predicted state estimate
        self.pf = np.dot(np.dot(Phi, self.pf), Phi.T) + Q  # Predicted state covariance

        # Measurement matrix
        H = np.eye(3, 6)
        
        # Update step
        Z = np.array([[M_rng1], [M_az], [M_el]])  # Measurement vector
        Inn = Z - np.dot(H, self.Sf)  # Innovation
        S = np.dot(H, np.dot(self.pf, H.T)) + self.R  # Innovation covariance
        K = np.dot(np.dot(self.pf, H.T), np.linalg.inv(S))  # Kalman gain
        self.Sf = self.Sf + np.dot(K, Inn)  # Updated state estimate
        self.pf = np.dot(np.eye(6) - np.dot(K, H), self.pf)  # Updated state covariance
        
        # Return the predicted state
        return self.Sf


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
    # Add more measurements as needed
]

# Process measurements and get predicted state estimates at each time step
for measurement in measurements:
    M_rng1, M_az, M_el, M_time = measurement
    predicted_state = kalman_filter.Filter_state_covariance(M_rng1, M_az, M_el, M_time)
    predicted_range = predicted_state[0, 0]  # Predicted range
    predicted_azimuth = predicted_state[1, 0]  # Predicted azimuth
    predicted_elevation = predicted_state[2, 0]  # Predicted elevation
    predicted_time = M_time  # Predicted time (same as measured time)
    filtered_state_fx = kalman_filter.Sf[0, 0]  # Filtered state fx
    filtered_state_fy = kalman_filter.Sf[1, 0]  # Filtered state fy
    filtered_state_fz = kalman_filter.Sf[2, 0]  # Filtered state fz
    print("Predicted range:", predicted_range)
    print("Predicted azimuth:", predicted_azimuth)
    print("Predicted elevation:", predicted_elevation)
    print("Predicted time:", predicted_time)
    print("Filtered state fx:", filtered_state_fx)
    print("Filtered state fy:", filtered_state_fy)
    print("Filtered state fz:", filtered_state_fz)
