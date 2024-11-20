import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepONet(nn.Module):
    def __init__(self, branch_input_dim, trunk_input_dim, hidden_dim=100, layers=7):
        super(DeepONet, self).__init__()
        self.branch_net = nn.Sequential(
            nn.Linear(branch_input_dim, hidden_dim),
            nn.ReLU(),
            *[
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU())
                for _ in range(layers - 1)
            ],
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.trunk_net = nn.Sequential(
            nn.Linear(trunk_input_dim, hidden_dim),
            nn.ELU(),
            *[
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
                for _ in range(layers - 1)
            ],
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, branch_input, x, t):
        branch_out = self.branch_net(branch_input)
        trunk_input = torch.concat((x, t), dim=1)
        trunk_out = self.trunk_net(trunk_input)
        output = torch.sum(branch_out * trunk_out, dim=-1)
        return output


# Supervised Loss
def data_fit_loss(model, ic, ic_sampled, x_data, t_data):
    Ns = 200
    B = ic.shape[0]

    x = x_data.unsqueeze(1).repeat(1, B).transpose(0, 1)

    x_ic, x_idx = sample_ics(x, Ns)
    t_ic = torch.zeros((ic.shape[0], Ns), device=device)

    ic_s = ic_sampled.unsqueeze(2).repeat(1, 1, Ns).transpose(1, 2)
    x_ic = x_ic.unsqueeze(2)
    t_ic = t_ic.unsqueeze(2)

    ic_s = ic_s.reshape(B * Ns, -1)
    x_ic = x_ic.reshape(B * Ns, -1)
    t_ic = t_ic.reshape(B * Ns, -1)

    # Model prediction
    u_pred_ic = model(ic_s, x_ic, t_ic)
    u_pred_ic = u_pred_ic.reshape(B, Ns)

    x_ic = x_ic.reshape(B, Ns)
    u_actual_ic = ic[0, :][x_idx]

    # boundary condition
    Nb = 100
    Lx = 6.0
    B = ic.shape[0]

    t = t_data.unsqueeze(1).repeat(1, B).transpose(0, 1)

    t_bc, t_idx = sample_ics(t, Ns)
    x_bc1 = torch.zeros((ic.shape[0], Ns), device=device)
    x_bc2 = torch.ones((ic.shape[0], Ns), device=device) * Lx

    x_bc1 = x_bc1.unsqueeze(2)
    x_bc2 = x_bc2.unsqueeze(2)
    t_bc = t_bc.unsqueeze(2)

    x_bc1 = x_bc1.reshape(B * Ns, -1)
    x_bc2 = x_bc2.reshape(B * Ns, -1)
    t_bc = t_bc.reshape(B * Ns, -1)

    # Model prediction
    u_pred_bc1 = model(ic_s, x_bc1, t_bc)
    u_pred_bc2 = model(ic_s, x_bc2, t_bc)

    return torch.mean((u_pred_ic - u_actual_ic) ** 2) + torch.mean(
        (u_pred_bc2 - u_pred_bc1) ** 2
    )


def pinn_loss(model, ic, x, t, mu):
    Lx = 6.0
    Tf = 16.0
    Nr = 300
    B = ic.shape[0]

    x = torch.rand((ic.shape[0], Nr), device=device) * Lx + 1e-6
    t = torch.rand((ic.shape[0], Nr), device=device) * Tf + 1e-6

    x.requires_grad_(True)
    t.requires_grad_(True)

    ic = ic.unsqueeze(2).repeat(1, 1, Nr).transpose(1, 2)
    x = x.unsqueeze(2)
    t = t.unsqueeze(2)

    ic = ic.reshape(B * Nr, -1)
    x = x.reshape(B * Nr, -1)
    t = t.reshape(B * Nr, -1)

    # Model prediction
    u_pred = model(ic, x, t)

    # Compute partial derivatives
    u_t = torch.autograd.grad(
        u_pred, t, grad_outputs=torch.ones_like(u_pred), create_graph=True
    )[0]
    u_x = torch.autograd.grad(
        u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True
    )[0]
    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0]

    # Diffusion equation residual
    residual = u_t - mu * u_xx
    return torch.mean(residual**2)


def sample_ics(input_tensor, N):
    # TODO: Don't repeat sensor points
    batch_size, num_columns = input_tensor.shape
    random_indices = torch.randint(0, num_columns, (batch_size, N), device=device)

    # Sort the random indices along the last dimension
    sorted_indices, _ = torch.sort(random_indices, dim=1)

    # Use advanced indexing to gather the sampled values
    result = torch.gather(input_tensor, 1, sorted_indices)
    return result, sorted_indices


def train(hdf5_file):
    # Parameters
    mu = 0.01  # Diffusion coefficient
    epochs = 5000
    lr = 1e-3
    branch_input_dim = 200  # For u0(x) - inputs sensor locations
    trunk_input_dim = 2  # For x, t
    alpha, beta = 150, 20

    # Initialize the model
    model = DeepONet(branch_input_dim, trunk_input_dim)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Load HDF5 data
    with h5py.File(hdf5_file, "r") as f:
        soln_data = torch.tensor(
            np.array(f["soln"]), dtype=torch.float32, device=device
        )
        t_data = torch.tensor(f["time"], dtype=torch.float32, device=device)
        x_data = torch.tensor(f["x"], dtype=torch.float32, device=device)

    train_data = soln_data[:200]
    val_data = soln_data[200:300]
    test_data = soln_data[300:500]

    ic_data = train_data[:, 0, :]

    ic_data_sampled, _ = sample_ics(ic_data, branch_input_dim)

    # Training loop
    for epoch in range(epochs):
        optimizer.zero_grad()

        # Physics-Informed Loss
        pinn_loss_value = pinn_loss(model, ic_data_sampled, x_data, t_data, mu)

        # Supervised Loss
        data_fit_loss_value = data_fit_loss(
            model, ic_data, ic_data_sampled, x_data, t_data
        )

        # Combine losses
        total_loss = alpha * pinn_loss_value + beta * data_fit_loss_value

        total_loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(
                f"Epoch {epoch}/{epochs}, Total Loss: {total_loss.item():.6f}, "
                f"PINN Loss: {pinn_loss_value.item():.6f}, Data fit Loss: {data_fit_loss_value.item():.6f}"
            )

    print("Training complete.")
    return model


if __name__ == "__main__":
    dataset = (
        "/scratch/venkvis_root/venkvis/shared_data/symmetry/data/dataset/heat_new.hdf5"
    )
    train(dataset)
