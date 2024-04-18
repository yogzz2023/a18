import numpy as np

class CVFilter:
    def __init__(self):
        self.Sf = np.zeros((6, 1))
        self.Filtered_Time = 0  # You need to define Filtered_Time
        self.pf = np.zeros((6, 6))
        self.R = np.zeros((3, 3))  # Assuming R is a 3x3 matrix
        self.sig_e_sqr = 0  # You need to define sig_e_sqr
        self.sig_r = 0  # You need to define sig_r
        self.sig_a = 0  # You need to define sig_a
        self.r = 0  # You need to define r
        self.plant_noise = 20  # Provided plant noise value
        self.H = np.zeros((1, 6))  # Define the H matrix (size 1x6)
        self.Z = 0  # Define the Z vector (size 1x1)
        self.Meas_Time = 0  # You need to define Meas_Time

    def Initialize_Filter_state_covariance(self, x, y, z, vx, vy, vz, time):
        self.Sf[0, 0] = x
        self.Sf[1, 0] = y
        self.Sf[2, 0] = z
        self.Sf[3, 0] = vx
        self.Sf[4, 0] = vy
        self.Sf[5, 0] = vz
        self.Filtered_Time = time
        for i in range(6):
            for j in range(6):
                k = i % 3
                l = j % 3
                self.pf[i, j] = self.R[k, l]

    def predict_state_covariance(self, float_det):
        Phi = np.zeros((6, 6))
        Q = np.zeros((6, 6))
        T_3 = (self.delt * self.delt * self.delt) / 3.0
        T_2 = (self.delt * self.delt) / 2.0

        Phi[0, 0] = 1
        Phi[0, 3] = self.delt
        Phi[1, 1] = 1
        Phi[1, 4] = self.delt
        Phi[2, 2] = 1
        Phi[2, 5] = self.delt
        Phi[3, 3] = 1
        Phi[4, 4] = 1
        Phi[5, 5] = 1

        self.Sp = np.dot(Phi, self.sf)
        self.predicted_time = self.Filtered_Time + self.delt

        Q[0, 0] = T_3
        Q[1, 1] = T_3
        Q[2, 2] = T_3
        Q[0, 3] = T_2
        Q[1, 4] = T_2
        Q[2, 5] = T_2
        Q[3, 0] = T_2
        Q[4, 1] = T_2
        Q[5, 2] = T_2
        Q[3, 3] = self.delt
        Q[4, 4] = self.delt
        Q[5, 5] = self.delt

        Q *= self.plant_noise
        self.Pp = np.dot(Phi, np.dot(self.pf, Phi.T)) + Q

    def Filter_state_covariance(self):
        Prev_Sf = self.Sf.copy()
        Prev_Filtered_Time = self.Filtered_Time
        S = self.R + np.dot(self.H, np.dot(self.Pp, self.H.T))
        K = np.dot(self.Pp, np.dot(self.H.T, np.linalg.inv(S)))
        Inn = self.Z - np.dot(self.H, self.Sp)
        self.Sf = self.Sp + np.dot(K, Inn)
        self.pf = np.dot(Inn - np.dot(K, self.H), self.Pp)
        self.Filtered_Time = self.Meas_Time
