import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
import wandb
from heat_symm import DeepONet, sample_ics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
wandb.init(project="pinn-symmetry", entity="pinn-symmetry", mode="disabled")

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def make_plot(
    model, model_sym, data, x, t, p_idxs=[0], dataset_name="train", branch_input_dim=200
):
    ic_data = data[:, 0, :]
    ic_data_sampled, _ = sample_ics(ic_data, branch_input_dim)

    for p_idx in p_idxs:
        ic0 = ic_data_sampled[p_idx]
        grid_x, grid_t = torch.meshgrid((x, t))

        grid_x, grid_t = grid_x.ravel(), grid_t.ravel()
        ic0 = ic0.unsqueeze(-1).repeat(1, len(grid_x)).transpose(0, 1)
        grid_x, grid_t = grid_x.unsqueeze(-1), grid_t.unsqueeze(-1)

        output = model(ic0, grid_x, grid_t)
        output = output.reshape((len(x), len(t))).transpose(0, 1)

        output_sym = model_sym(ic0, grid_x, grid_t)
        output_sym = output_sym.reshape((len(x), len(t))).transpose(0, 1)

        fig, axs = plt.subplots(3, 1)
        axs[0].pcolormesh(
            x.cpu(),
            t.cpu(),
            output.detach().cpu(),
            cmap="RdBu_r",
            shading="gouraud",
            rasterized=True,
            clim=(-0.8, 0.8),
        )
        axs[0].set_title("PINN")
        axs[1].pcolormesh(
            x.cpu(),
            t.cpu(),
            output_sym.detach().cpu(),
            cmap="RdBu_r",
            shading="gouraud",
            rasterized=True,
            clim=(-0.8, 0.8),
        )
        axs[1].set_title("Symmetry")
        axs[2].pcolormesh(
            x.cpu(),
            t.cpu(),
            data[p_idx].detach().cpu(),
            cmap="RdBu_r",
            shading="gouraud",
            rasterized=True,
            clim=(-0.8, 0.8),
        )
        axs[2].set_title("True solution (Spectral solver)")
        plt.savefig(f"plots/{dataset_name}_{p_idx}.pdf")
        plt.clf()


if __name__ == "__main__":
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
    test_data = soln_data[300:500]

    ################################################################
    branch_input_dim = 200  # For u0(x) - inputs sensor locations
    trunk_input_dim = 2  # For x, t

    model = DeepONet(branch_input_dim, trunk_input_dim)
    model = model.to(device)
    model.load_state_dict(torch.load("models/test2.pt", weights_only=True))

    model_sym = DeepONet(branch_input_dim, trunk_input_dim)
    model_sym = model_sym.to(device)
    model_sym.load_state_dict(torch.load("models/test_symm_2.pt", weights_only=True))

    make_plot(
        model,
        model_sym,
        train_data,
        x_data,
        t_data,
        [0, 4, 5, 13, 10],
        dataset_name="train",
    )
    make_plot(
        model,
        model_sym,
        test_data,
        x_data,
        t_data,
        [0, 40, 59, 131, 100],
        dataset_name="test",
    )
    print("Done with plots!")
