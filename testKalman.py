from array_ import array, tensor_to_array
from kalman import KalmanFilter
import torch
from torch.utils.tensorboard import SummaryWriter



def generate_parameters(x_N: int, z_N: int, T: int):
    # parameters of the system
    F = torch.rand(x_N, x_N) / 10
    B = torch.rand(x_N, x_N) / 10
    H = torch.rand(z_N, x_N) / 10
    U = torch.rand(x_N, T)

    X_t = torch.rand(x_N, T)
    X_clear = X_t.clone()

    Z_t = torch.rand(z_N, T)
    Z_clear = Z_t.clone()

    return F, B, H, U, X_t, X_clear, Z_t, Z_clear


def system(F: torch.Tensor, B: torch.Tensor, H: torch.Tensor, U: torch.Tensor, X_t: torch.Tensor, X_clear: torch.Tensor, Z_t: torch.Tensor, Z_clear: torch.Tensor, x_N: int, z_N: int, T: int):
    # result with and without noise
    for t in range(1, T):
        X_clear[:, t] = torch.matmul(F, X_t[:, t-1]) + torch.matmul(B, U[:, t-1])
        X_t[:, t] =  X_clear[:, t] + torch.randn(x_N) / 10
        Z_clear[:, t] = torch.matmul(H, X_t[:, t])
        Z_t[:, t] = Z_clear[:, t] + torch.randn(z_N) / 10
    return X_clear, Z_clear, X_t, Z_t


if __name__ == "__main__":
    x_N, z_N, T = 3, 2, 100

    F, B, H, U, X_t, X_clear, Z_t, Z_clear = generate_parameters(x_N, z_N, T)

    X_clear, Z_clear, X_t, Z_t = system(F, B, H, U, X_t, X_clear, Z_t, Z_clear, x_N, z_N, T)

    # Tensors of results
    x_filtered = torch.zeros(x_N, T)
    z_filtered = torch.zeros(z_N, T)

    x_filtered[:, 0] = X_t[:, 0]
    z_filtered[:, 0] = Z_t[:, 0]

    # Defining parameters P, Q, R
    P = torch.eye(x_N) * 0.1
    Q = torch.eye(x_N) * 0.1
    R = torch.eye(z_N) * 0.1



    F_ = tensor_to_array(F)
    B_ = tensor_to_array(B)
    H_ = tensor_to_array(H)
    P_ = tensor_to_array(P)
    Q_ = tensor_to_array(Q)
    R_ = tensor_to_array(R)

    kalman = KalmanFilter(F_, B_, H_, P_, x_N, z_N, Q_, R_)

    for t in range(1, T):
        x, z = kalman.filter(tensor_to_array(x_filtered[:, t-1]), tensor_to_array(U[:, t-1]), tensor_to_array(Z_t[:, t]))
        x_filtered[:, t] = torch.tensor(x.transpose().values[0])
        z_filtered[:, t] = torch.tensor(z.transpose().values[0])

    # drawing filtered datas for X and Z
    writer = SummaryWriter('./logs/KalmanFilter')

    # X_clear, X_t, x_filtered
    # Z_clear, Z_t, z_filtered
    for i in range(Z_clear.shape[0]):
        for t in range(Z_clear.shape[1]):
            writer.add_scalars('Z_Trajectory {}'.format(i), {'Clear': Z_clear[i, t], 'Noisy': Z_t[i, t], 'Filtered': z_filtered[i, t]}, t)

    for i in range(X_clear.shape[0]):
        for t in range(X_clear.shape[1]):
            writer.add_scalars('X_Trajectory {}'.format(i), {'Clear': X_clear[i, t], 'Noisy': X_t[i, t], 'Filtered': x_filtered[i, t]}, t)