from array_ import array


class KalmanFilter:
    def __init__(self, F: array, B: array, H: array, P: array, x_N: int, z_N: int, Q: array, R: array):
        # dimensions        
        self.x_N = x_N
        self.z_N = z_N

        # parameters of Prediction stage:
        self.x = None
        self.F = F
        self.B = B
        self.P = P
        self.Q = array.eye(self.x_N) * Q

        # parameters of Update stage:
        self.H = H
        self.R = array.eye(self.z_N) * R

    def _predict(self, x_k_1_k_1: array, u_k_1: array):
        self.x = array.__matmul__(self.F, x_k_1_k_1.transpose()) + array.__matmul__(self.B, u_k_1.transpose())
        self.P = array.__matmul__(array.__matmul__(self.F, self.P), self.F.transpose()) + self.Q

    def _update(self, z_k: array):
        S = array.__matmul__(array.__matmul__(self.H, self.P), self.H.transpose()) + self.R
        K = array.__matmul__(array.__matmul__(self.P, self.H.transpose()), array.inverse(S))
        self.x = self.x + array.__matmul__(K, (z_k.transpose() + array.__matmul__(self.H, self.x) * (-1)))
        self.P = self.P + array.__matmul__(array.__matmul__(K, self.H), self.P) * (-1)
        z = array.__matmul__(self.H, self.x)
        return self.x, z
    
    def filter(self, x_k_1_k_1: array, u_k_1: array, z_k: array):
        self._predict(x_k_1_k_1, u_k_1)
        return self._update(z_k)