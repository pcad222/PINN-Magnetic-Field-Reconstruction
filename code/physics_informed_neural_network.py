import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import trange
from torch.optim.lr_scheduler import StepLR

CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parent
DATA_DIR = ROOT_DIR / "data" / "uniform_field"
RESULTS_DIR = ROOT_DIR / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def normalization(B_fields):
    """normalzie each component using its mean and standard deviation as bz is like 1000 than bx and by"""
    mean = torch.mean(B_fields, dim=0)
    std = torch.std(B_fields, dim=0)
    norm = (B_fields - mean) / std
    return mean, std, norm


def collocation_df(df, radius=25, half_height=15, dr=1, dz=5, dazim=10):

    """
    Generate cylindrical collocation points, convert them to Cartesian coordinates,
    and return a DataFrame after removing duplicates and measurement-point overlaps """
    # 
    radii = np.arange(0, radius + 1e-12, dr)
    theta = np.arange(0, 360, dazim)
    z_values = np.arange(-half_height, half_height + 1e-12, dz)

    r, theta_grid, z = np.meshgrid(radii, theta, z_values, indexing="ij")
    x = r * np.cos(np.radians(theta_grid))
    y = r * np.sin(np.radians(theta_grid))

    colloc_df = pd.DataFrame({"x": x.ravel() / 100, "y": y.ravel() / 100, "z": z.ravel() / 100})
    colloc_df["key"] = list(map(tuple, colloc_df[["x", "y", "z"]].round(6).values))
    mapper_keys = set(map(tuple, df[["x", "y", "z"]].round(6).values))

    n_initial = len(colloc_df)
    colloc_df = colloc_df.drop_duplicates("key")
    n_unique = len(colloc_df)
    colloc_df = colloc_df[~colloc_df["key"].isin(mapper_keys)].drop(columns="key").reset_index(drop=True)

    colloc_info = pd.DataFrame(
        [{
            "radius_cm": radius, "height_cm": 2 * half_height, "dr_cm": dr, "dz_cm": dz,
            "dazim_degree": dazim, "rings": np.count_nonzero(radii), "z_planes": len(z_values),
            "azim_per_ring": len(theta), "duplicates_removed": n_initial - n_unique,
            "points": len(colloc_df),
        }]
    )

    return colloc_df, colloc_info 


class PINN(nn.Module):
    def __init__(self, hidden_layers, activation_fn):
        super().__init__()
        layers = []
        input_dim = 3 # x, y, z 

        for hidden_dim in hidden_layers:
            layers.extend([nn.Linear(input_dim, hidden_dim), activation_fn.__class__()])
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


MeanSqError = nn.MSELoss()

threshold_physics = 1e-2
NumofEpoches = 5000

lambda_data = 1
lambda_div = 1
lambda_curl = 1

num_neuron = [64, 32]

SEED = 42


def pinn_training(xyz_mapper, B_fields, mean_B, std_B, colloc_df):
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    coll_xyz = torch.tensor(colloc_df[["x", "y", "z"]].values, dtype=torch.float32, requires_grad=True)

    model = PINN(num_neuron, nn.Tanh())
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-7)
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)

    loss_history = {"total": [], "mse": [], "div": [], "curl": []}
    start_time = time.time()

    for _ in trange(NumofEpoches, desc="PINN training"):
        model.train()
        optimizer.zero_grad()
        coll_xyz.grad = None

        B_pred = model(xyz_mapper) * std_B + mean_B # denormalize as fields have been normalzied 
        B_true = B_fields * std_B + mean_B
        mse_loss = MeanSqError(B_pred, B_true)

        div_loss_val = 0.0
        curl_loss_val = 0.0

        if mse_loss.item() >= threshold_physics: # fit the data only  when data loss is high
            total_loss = lambda_data * mse_loss

        else:
            B_out = model(coll_xyz) * std_B + mean_B 
            #    # Denormalize predicted fields at  collocation, and compute del.B and del cross B losses.
            Bx, By, Bz = B_out[:, 0], B_out[:, 1], B_out[:, 2]

            grad_Bx = torch.autograd.grad(Bx.sum(), coll_xyz, create_graph=True)[0]
            grad_By = torch.autograd.grad(By.sum(), coll_xyz, create_graph=True)[0]
            grad_Bz = torch.autograd.grad(Bz.sum(), coll_xyz, create_graph=True)[0]

            div_B = grad_Bx[:, 0] + grad_By[:, 1] + grad_Bz[:, 2]

            curl_x = grad_Bz[:, 1] - grad_By[:, 2]
            curl_y = grad_Bx[:, 2] - grad_Bz[:, 0]
            curl_z = grad_By[:, 0] - grad_Bx[:, 1]

            div_loss = torch.mean(div_B**2)
            curl_loss = torch.mean(curl_x**2 + curl_y**2 + curl_z**2)

            div_loss_val = div_loss.item()
            curl_loss_val = curl_loss.item()

            total_loss = lambda_data * mse_loss + lambda_div * div_loss + lambda_curl * curl_loss

        total_loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history["total"].append(total_loss.item())
        loss_history["mse"].append(mse_loss.item())
        loss_history["div"].append(div_loss_val)
        loss_history["curl"].append(curl_loss_val)

    model_details = {
        "seed": SEED,
        "collocation_points": len(coll_xyz),
        "architecture": str(num_neuron),
        "activation": "Tanh",
        "parameters": sum(p.numel() for p in model.parameters()),
        "lambda_data": lambda_data,
        "lambda_div": lambda_div,
        "lambda_curl": lambda_curl,
        "time_seconds": time.time() - start_time,
        "final_total": loss_history["total"][-1],
        "final_mse": loss_history["mse"][-1],
        "final_div": loss_history["div"][-1],
        "final_curl": loss_history["curl"][-1],
    }

    return model, loss_history, pd.DataFrame([model_details])


def evaluate_mapper_points(model, mapper_df, mean_B, std_B): # makes predition after training 
    xyz_cols = ["x", "y", "z"]
    B_cols = ["bx", "by", "bz"]

    grad_cols = [
        "dBx_dx", "dBx_dy", "dBx_dz",
        "dBy_dx", "dBy_dy", "dBy_dz",
        "dBz_dx", "dBz_dy", "dBz_dz",
    ]

    xyz_np = mapper_df[xyz_cols].to_numpy()
    B_true = mapper_df[B_cols].to_numpy()

    device = next(model.parameters()).device

    xyz = torch.tensor(xyz_np, dtype=torch.float32, device=device, requires_grad=True)

    model.eval()

    B = model(xyz) * std_B.to(device) + mean_B.to(device) # denormalzied as field were normalzied 

    gradients = [torch.autograd.grad(B[:, k].sum(), xyz, retain_graph=(k < 2))[0] for k in range(3)]

    B_pred = B.detach().cpu().numpy()

    jac = torch.stack(gradients, dim=1).detach().cpu().numpy().reshape(-1, 9)

    residual = B_pred - B_true

    prediction_df = pd.DataFrame(np.c_[xyz_np, B_pred], columns=xyz_cols + B_cols)

    gradient_df = pd.DataFrame(jac, columns=grad_cols)

    dBx_dx = gradient_df["dBx_dx"]
    dBx_dy = gradient_df["dBx_dy"]
    dBx_dz = gradient_df["dBx_dz"]

    dBy_dx = gradient_df["dBy_dx"]
    dBy_dy = gradient_df["dBy_dy"]
    dBy_dz = gradient_df["dBy_dz"]

    dBz_dx = gradient_df["dBz_dx"]
    dBz_dy = gradient_df["dBz_dy"]
    dBz_dz = gradient_df["dBz_dz"]

    div_B = dBx_dx + dBy_dy + dBz_dz

    curl_x = dBz_dy - dBy_dz
    curl_y = dBx_dz - dBz_dx
    curl_z = dBy_dx - dBx_dy

    curl_mag = np.sqrt(curl_x**2 + curl_y**2 + curl_z**2)

    metrics_df = pd.DataFrame([{
        "div_B_mean": div_B.mean(),
        "div_B_std": div_B.std(),
        "div_B_rms": np.sqrt(np.mean(div_B**2)),

        "curl_x_mean": curl_x.mean(),
        "curl_x_std": curl_x.std(),
        "curl_x_rms": np.sqrt(np.mean(curl_x**2)),

        "curl_y_mean": curl_y.mean(),
        "curl_y_std": curl_y.std(),
        "curl_y_rms": np.sqrt(np.mean(curl_y**2)),

        "curl_z_mean": curl_z.mean(),
        "curl_z_std": curl_z.std(),
        "curl_z_rms": np.sqrt(np.mean(curl_z**2)),

        "curl_mag_mean": curl_mag.mean(),
        "curl_mag_std": curl_mag.std(),
        "curl_mag_rms": np.sqrt(np.mean(curl_mag**2)),

        "bx_mean": residual[:, 0].mean(),
        "bx_std": residual[:, 0].std(),
        "bx_rmse": np.sqrt(np.mean(residual[:, 0]**2)),

        "by_mean": residual[:, 1].mean(),
        "by_std": residual[:, 1].std(),
        "by_rmse": np.sqrt(np.mean(residual[:, 1]**2)),

        "bz_mean": residual[:, 2].mean(),
        "bz_std": residual[:, 2].std(),
        "bz_rmse": np.sqrt(np.mean(residual[:, 2]**2)),

        "field_rmse": np.sqrt(np.mean(residual**2)),
    }])

    gradient_mean_std_df = pd.DataFrame({
        "mean": gradient_df.mean(),
        "std": gradient_df.std(),
        "rms": np.sqrt((gradient_df**2).mean()),
    })

    return prediction_df, gradient_df, gradient_mean_std_df, metrics_df


def save_results(filename, **results):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)

    with open(filename, "wb") as file:
        pickle.dump(results, file)


def load_results(filename):
    with open(filename, "rb") as file:
        return pickle.load(file)


def run_uniform_field(n_points):
    data_file = DATA_DIR / f"uniform_field_{n_points}_points.csv"
   # print(f"Loading: {data_file}")

    mapper_df = pd.read_csv(data_file)
    mapper_df.columns = ["x", "y", "z", "bx", "by", "bz"]

    xyz_mapper = torch.tensor(mapper_df[["x", "y", "z"]].values, dtype=torch.float32)
    mapper_field = torch.tensor(mapper_df[["bx", "by", "bz"]].values, dtype=torch.float32)

    mean_B, std_B, B_fields = normalization(mapper_field)

    normalization_df = pd.DataFrame(
        [
            mean_B.detach().cpu().numpy(),
            std_B.detach().cpu().numpy(),
        ],
        index=["mean", "std"],
        columns=["bx", "by", "bz"],)

    colloc_df, colloc_info = collocation_df(mapper_df, dr=1)

    model, loss_history, model_details_df = pinn_training( xyz_mapper=xyz_mapper, B_fields=B_fields, mean_B=mean_B,
        std_B=std_B, colloc_df=colloc_df,)

    (
        prediction_df, gradient_df, gradient_mean_std_df,  metrics_df, ) = evaluate_mapper_points(
        model=model, mapper_df=mapper_df, mean_B=mean_B,std_B=std_B, )

    case_dir = RESULTS_DIR / f"pinn_{n_points}" # computed values saved at the foilder results for each nnumber of points
    case_dir.mkdir(parents=True, exist_ok=True)

    save_path = case_dir / "results.pkl"

    save_results(
        save_path,
        n_points=n_points,
        mapper_df=mapper_df,
        normalization_df=normalization_df,
        colloc_df=colloc_df,
        colloc_info=colloc_info,
        loss_history=loss_history,
        model_details_df=model_details_df,
        prediction_df=prediction_df,
        gradient_df=gradient_df,
        linear_gradient=gradient_mean_std_df,
        metrics_df=metrics_df,
    )

    torch.save(model.state_dict(), case_dir / "model.pt")

    print(f"Saved results: {save_path}")
   # print(f"Saved model:   {case_dir / 'model.pt'}")
