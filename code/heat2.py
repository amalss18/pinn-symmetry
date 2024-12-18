import torch
import torch.nn as nn
import torch.optim as optim
import h5py
import numpy as np
import wandb
import matplotlib.pyplot as plt
import argparse


class DeepONet(nn.Module):
    def __init__(self, branch_input_dim, trunk_input_dim, hidden_dim=100, layers=7):
        super(DeepONet, self).__init__()
        self.branch_net = nn.Sequential(
            nn.Linear(branch_input_dim, hidden_dim),
            nn.ELU(),
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
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ELU())
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

    ic_s = ic_s.reshape(B * Ns, -1)
    x_ic = x_ic.reshape(B * Ns, -1)
    t_ic = t_ic.reshape(B * Ns, -1)

    # Model prediction
    u_pred_ic = model(ic_s, x_ic, t_ic)
    u_pred_ic = u_pred_ic.reshape(B, Ns)

    x_ic = x_ic.reshape(B, Ns)
    u_actual_ic = torch.gather(ic, dim=1, index=x_idx)

    # boundary condition
    Nb = 100
    Lx = 6.0
    B = ic.shape[0]

    ic_s_bc = ic_sampled.unsqueeze(2).repeat(1, 1, Nb).transpose(1, 2)
    t = t_data.unsqueeze(1).repeat(1, B).transpose(0, 1)

    t_bc, t_idx = sample_ics(t, Nb)
    x_bc1 = torch.zeros((ic.shape[0], Nb), device=device)
    x_bc2 = torch.ones((ic.shape[0], Nb), device=device) * Lx

    ic_s = ic_s_bc.reshape(B * Nb, -1)
    x_bc1 = x_bc1.reshape(B * Nb, -1)
    x_bc2 = x_bc2.reshape(B * Nb, -1)
    t_bc = t_bc.reshape(B * Nb, -1)

    # Model prediction
    u_pred_bc1 = model(ic_s, x_bc1, t_bc)
    u_pred_bc2 = model(ic_s, x_bc2, t_bc)

    return torch.mean((u_pred_ic - u_actual_ic) ** 2) + torch.mean(
        (u_pred_bc2 - u_pred_bc1) ** 2
    ), torch.mean((u_pred_ic - u_actual_ic) ** 2)


def pinn_loss(model, ic, x, t, mu, Nr=300):
    Lx = 6.0
    Tf = 16.0
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


def compute_test_loss(model, data, x, t):
    ic_data = data[:, 0, :]
    ic, _ = sample_ics(ic_data, 200)

    loss = 0.0
    for d_idx, ic0 in enumerate(ic):
        grid_x, grid_t = torch.meshgrid((x[::8], t[::8]))

        grid_x, grid_t = grid_x.ravel(), grid_t.ravel()
        ic0 = ic0.unsqueeze(-1).repeat(1, len(grid_x)).transpose(0, 1)
        grid_x, grid_t = grid_x.unsqueeze(-1), grid_t.unsqueeze(-1)

        output = model(ic0, grid_x, grid_t)
        output = output.reshape((len(x[::8]), len(t[::8]))).transpose(0, 1)

        true_output = data[d_idx]
        loss += torch.mean((output - true_output[::8, ::8]) ** 2)

    return loss / data.shape[0]


def train(train_data, val_data, test_data, x_data, t_data, Nr=300):
    # Parameters
    mu = 0.01  # Diffusion coefficient
    epochs = 10000
    lr = 1e-3
    branch_input_dim = 200  # For u0(x) - inputs sensor locations
    trunk_input_dim = 2  # For x, t
    alpha, beta = 150, 20

    # Initialize the model
    model = DeepONet(branch_input_dim, trunk_input_dim)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    ic_data = train_data[:, 0, :]
    ic_data_sampled, _ = sample_ics(ic_data, branch_input_dim)

    # Training loop
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # Physics-Informed Loss
        pinn_loss_value = pinn_loss(model, ic_data_sampled, x_data, t_data, mu, Nr=Nr)

        # Supervised Loss
        data_fit_loss_value, ic_loss_value = data_fit_loss(
            model, ic_data, ic_data_sampled, x_data, t_data
        )

        # Combine losses
        total_loss = alpha * pinn_loss_value + beta * data_fit_loss_value
        wandb.log(
            {
                "pinn_loss": pinn_loss_value,
                "data_fit_loss": data_fit_loss_value,
                "ic_loss": ic_loss_value,
                "total_loss": total_loss,
                "epoch": epoch,
            }
        )

        total_loss.backward()
        optimizer.step()

        ################### validation #########################################
        model.eval()

        ic_data_val = val_data[:, 0, :]
        ic_data_val_sampled, _ = sample_ics(ic_data_val, branch_input_dim)
        pinn_loss_val = pinn_loss(model, ic_data_val_sampled, x_data, t_data, mu, Nr)

        # Supervised Loss
        data_fit_loss_val, ic_loss_val = data_fit_loss(
            model, ic_data_val, ic_data_val_sampled, x_data, t_data
        )

        # Combine losses
        val_loss = alpha * pinn_loss_val + beta * data_fit_loss_val
        wandb.log(
            {
                "val_loss": val_loss,
                "epoch": epoch,
            }
        )

        if epoch % 500 == 0:
            test_loss = compute_test_loss(model, test_data, x_data, t_data)
            print(f"Epoch {epoch}/{epochs}, Test Loss: {test_loss.item():.6f}")
            wandb.log(
                {
                    "test_loss": test_loss,
                    "epoch": epoch,
                }
            )
            print(
                f"Epoch {epoch}/{epochs}, Total Loss: {total_loss.item():.6f}, "
                f"PINN Loss: {pinn_loss_value.item():.6f}, Data fit Loss: {data_fit_loss_value.item():.6f}"
            )

    print("Training complete.")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--Nr", type=int, default=500)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wandb.init(
        project="pinn-symmetry", entity="pinn-symmetry", mode="online", name=args.name
    )

    torch.manual_seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dataset = (
        "/scratch/venkvis_root/venkvis/shared_data/symmetry/data/dataset/heat_new.hdf5"
    )
    # Load HDF5 data
    with h5py.File(dataset, "r") as f:
        soln_data = torch.tensor(
            np.array(f["soln"]), dtype=torch.float32, device=device
        )
        t_data = torch.tensor(f["time"], dtype=torch.float32, device=device)
        x_data = torch.tensor(f["x"], dtype=torch.float32, device=device)

    train_data = soln_data[:200]
    val_data = soln_data[200:300]
    test_data = soln_data[300:350]

    model = train(train_data, val_data, test_data, x_data, t_data, Nr=args.Nr)
    torch.save(model.state_dict(), f"models/heat_{args.name}.pt")
