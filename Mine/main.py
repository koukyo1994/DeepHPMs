import sys
import numpy as np
import scipy.io
import argparse
from pyDOE import lhs
from scipy.interpolate import griddata
from core.plot import plt_saver

if __name__ == "__main__":
    sys.path.append("./")
    from core.model import DeepHPM

    parser = argparse.ArgumentParser()
    parser.add_argument("--niter", default=10000, type=int)
    parser.add_argument("--scipyopt", default=False)
    args = parser.parse_args()

    # Domain Boundary
    idn_lb = np.array([0.0, -8.0])
    idn_ub = np.array([10.0, 8.0])

    sol_lb = np.array([0.0, -8.0])
    sol_ub = np.array([10.0, 8.0])

    # Load Data
    idn_data = scipy.io.loadmat("../Data/burgers_sine.mat")
    idn_t = idn_data["t"].flatten()[:, None]
    idn_x = idn_data["x"].flatten()[:, None]
    idn_exact = np.real(idn_data["usol"])

    idn_T, idn_X = np.meshgrid(idn_t, idn_x)

    keep = 2 / 3
    index = int(keep * idn_t.shape[0])
    idn_T = idn_T[:, 0:index]
    idn_X = idn_X[:, 0:index]
    idn_exact = idn_exact[:, 0:index]

    idn_t_star = idn_T.flatten()[:, None]
    idn_x_star = idn_X.flatten()[:, None]
    idn_X_star = np.hstack([idn_t_star, idn_x_star])
    idn_u_star = idn_exact.flatten()[:, None]

    sol_data = scipy.io.loadmat("../Data/burgers.mat")

    sol_t = sol_data["t"].flatten()[:, None]
    sol_x = sol_data["x"].flatten()[:, None]
    sol_exact = np.real(sol_data["usol"])

    sol_T, sol_X = np.meshgrid(sol_t, sol_x)

    sol_t_star = sol_T.flatten()[:, None]
    sol_x_star = sol_X.flatten()[:, None]
    sol_X_star = np.hstack([sol_t_star, sol_x_star])
    sol_u_star = sol_exact.flatten()[:, None]

    # Training Data
    N_train = 10000
    idx = np.random.choice(idn_t_star.shape[0], N_train, replace=False)
    train_t = idn_t_star[idx, :]
    train_x = idn_x_star[idx, :]
    train_u = idn_u_star[idx, :]

    noise = 0.0
    train_u = train_u + noise * np.std(train_u) * np.random.randn(
        train_u.shape[0], train_u.shape[1])

    # Solution Data
    N0 = sol_exact.shape[0]
    N_b = sol_exact.shape[1]
    N_f = 20000

    idx_x = np.random.choice(sol_x.shape[0], N0, replace=False)
    train_x0 = sol_x[idx_x, :]
    train_u0 = sol_exact[idx_x, 0:1]

    idx_t = np.random.choice(sol_t.shape[0], N_b, replace=False)
    train_tb = sol_t[idx_t, :]

    train_X_f = sol_lb + (sol_ub - sol_lb) * lhs(2, N_f)

    # Layers
    u_layers = [2, 50, 50, 50, 50, 1]
    pde_layers = [3, 100, 100, 1]
    layers = [2, 50, 50, 50, 50, 1]

    # Model
    model = DeepHPM(idn_lb, idn_ub, train_t, train_x, train_u, train_tb,
                    train_x0, train_u0, train_X_f, layers, sol_lb, sol_ub,
                    u_layers, pde_layers)

    # Train the Identifier
    model.train_u(args.niter, "model/burgers_saved.model", args.scipyopt)
    # model.train_f(args.niter, "model/burgers_saved.model", args.scipyopt)

    idn_u_pred, idn_f_pred = model.identifier_predict(idn_t_star, idn_x_star)
    idn_error_u = np.linalg.norm(idn_u_star - idn_u_pred, 2) / np.linalg.norm(
        idn_u_star, 2)
    model.logger.info(f"Error u: {idn_error_u:.3e}")

    model.train_solver(args.niter, args.scipyopt)
    u_pred, f_pred = model.solver_predict(sol_t_star, sol_x_star)
    error_u = np.linalg.norm(sol_u_star - u_pred, 2) / np.linalg.norm(
        sol_u_star, 2)
    model.logger.info(f"Error u: {error_u:.3e}")

    U_pred = griddata(
        sol_X_star, u_pred.flatten(), (sol_T, sol_X), method="cubic")
    plt_saver(U_pred, sol_exact, sol_lb, sol_ub, "Burgers_different_simultaneous")
