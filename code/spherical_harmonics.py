import numpy as np
import pandas as pd
from pathlib import Path
from sympy import symbols, sqrt, acos, atan2, cos, sin, Piecewise, assoc_legendre, factorial, diff, lambdify

CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parent
DATA_DIR = ROOT_DIR / "data" / "uniform_field"

l_max = 3


def build_pi_design_matrix(df, l_max=l_max):
    # measured, simulated, 3d  coordinates
    x_coord = df["x"].to_numpy(dtype=float)
    y_coord = df["y"].to_numpy(dtype=float)
    z_coord = df["z"].to_numpy(dtype=float)

    # ---- symbolic coords ----
    x, y, z = symbols("x y z", real=True)
    r = sqrt(x**2 + y**2 + z**2); phi = atan2(y, x); theta = acos(z / (r + 1e-12)) # 1e-12,  offset, prevents division by zero at the origin

    pi_x, pi_y, pi_z = [], [], []
    l_val, m_val = [], []

    for l in range(l_max + 1):
        lp1 = l + 1 # shiting l to l+1, so that l= 0, becomes l, and has  m = (-1, 0,1 )  for bx, by and bz
        for m in range(-(l + 1), l + 2):  # m = -(l+1)..(l+1)
            abs_m = abs(m)

            pref = factorial(lp1 - 1) * (-2)**abs_m / factorial(lp1 + abs_m)
            if m >= 0:
                C = pref * cos(m * phi)
            else:
                C = pref * sin(abs_m * phi)



            Sigma_lp1_m = C * r**lp1 * assoc_legendre(lp1, abs_m, cos(theta))

            bx_fun = lambdify((x, y, z), diff(Sigma_lp1_m, x), "numpy")
            by_fun = lambdify((x, y, z), diff(Sigma_lp1_m, y), "numpy")
            bz_fun = lambdify((x, y, z), diff(Sigma_lp1_m, z), "numpy")

            pi_x.append(bx_fun(x_coord, y_coord, z_coord))
            pi_y.append(by_fun(x_coord, y_coord, z_coord))
            pi_z.append(bz_fun(x_coord, y_coord, z_coord))

            l_val.append(l); m_val.append(m)

    lm_df = pd.DataFrame({"l": l_val, "m": m_val})
    pi_x = np.column_stack(pi_x); pi_y = np.column_stack(pi_y); pi_z = np.column_stack(pi_z)

    return np.vstack([pi_x, pi_y, pi_z]), lm_df


def compute_coeff(df, basis_matrix, lm_df):
    bx, by, bz = [df[col].to_numpy() for col in ('bx', 'by', 'bz')]
    field = np.hstack((bx, by, bz))
    coeffs = np.linalg.lstsq(basis_matrix, field, rcond=None)[0]

    glm_df = lm_df.copy()
    glm_df['coeffs'] = coeffs

    pred_df = df[['x', 'y', 'z']].copy()
    pi_x, pi_y, pi_z = np.split(basis_matrix, 3)

    pred_df['bx'] = np.matmul(pi_x, coeffs)
    pred_df['by'] = np.matmul(pi_y, coeffs)
    pred_df['bz'] = np.matmul(pi_z, coeffs)

    return pred_df, glm_df


def gradients_from_lm(lm_df): # need to use Maxwell equations del. B and del cross B to get 4 terms as sh only computes 5 terms
    g1m2 = lm_df.loc[(lm_df["l"]==1) & (lm_df["m"]==-2), "coeffs"].iloc[0]
    g1m1 = lm_df.loc[(lm_df["l"]==1) & (lm_df["m"]==-1), "coeffs"].iloc[0]
    g10  = lm_df.loc[(lm_df["l"]==1) & (lm_df["m"]==0), "coeffs"].iloc[0]
    g11  = lm_df.loc[(lm_df["l"]==1) & (lm_df["m"]==1), "coeffs"].iloc[0]
    g12  = lm_df.loc[(lm_df["l"]==1) & (lm_df["m"]==2), "coeffs"].iloc[0]

    derivative = np.array([[g12-0.5*g10, g1m2, g11, g1m2, -g12-0.5*g10, g1m1, g11, g1m1, g10]])

    return pd.DataFrame(derivative, columns=["dBx/dx", "dBx/dy", "dBx/dz", "dBy/dx", "dBy/dy", "dBy/dz", "dBz/dx", "dBz/dy", "dBz/dz"])


def spherical_harmonics(n_points, l_max=3):
    df = pd.read_csv(DATA_DIR / f"uniform_field_{n_points}_points.csv") # dataframe  based on for  each data point
    basis_matrix, lm_df = build_pi_design_matrix(df, l_max)
    pred_df, glm_df = compute_coeff(df, basis_matrix, lm_df)
    linear_gradient_df = gradients_from_lm(glm_df)

    return pred_df, glm_df, linear_gradient_df
