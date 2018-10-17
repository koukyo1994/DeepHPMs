import sys
import numpy as np
import argparse

from scipy.interpolate import griddata
from core.plot import plt_saver
from core.loader import DataLoader

if __name__ == "__main__":
    sys.path.append("./")
    from core.model import DeepHPM

    parser = argparse.ArgumentParser()
    parser.add_argument("--niter", default=0)
    parser.add_argument("--scipyopt", default=True)
    args = parser.parse_args()

    dataloader = DataLoader(
        ["../Data/burgers.mat", "../MyData/burgers_cos.mat"],
        "../MyData/burgers_sine.mat", 10.0, 8.0)
    sol_data = dataloader.get_solver_data(20000)
    idn_data1 = dataloader.get_train_data("../MyData/burgers_cos.mat")
    idn_data2 = dataloader.get_train_data("../Data/burgers.mat")

    u_layers = [2, 50, 50, 50, 50, 1]
    pde_layers = [3, 100, 100, 1]
    layers = [2, 50, 50, 50, 50, 1]

    idn_lb = idn_data1["idn_lb"]
    idn_ub = idn_data1["idn_ub"]
    train_t = idn_data1["train_t"]
    train_x = idn_data1["train_x"]
    train_u = idn_data1["train_u"]

    train_x0 = sol_data["train_x0"]
    train_u0 = sol_data["train_u0"]
    sol_lb = sol_data["sol_lb"]
    sol_ub = sol_data["sol_ub"]
    train_tb = sol_data["train_tb"]
    train_X_f = sol_data["train_X_f"]

    model = DeepHPM(idn_lb, idn_ub, train_t, train_x, train_u, train_tb,
                    train_x0, train_u0, train_X_f, layers, sol_lb, sol_ub,
                    u_layers, pde_layers)

    model.train_u(args.niter, "model/saved.model", args.scipyopt)
    model.train_f(args.niter, "model/saved.model", args.scipyopt)

    idn_lb2 = idn_data2["idn_lb"]
    idn_ub2 = idn_data2["idn_ub"]
    train_t2 = idn_data2["train_t"]
    train_x2 = idn_data2["train_x"]
    train_u2 = idn_data2["train_u"]

    model.logger.info("Change train data")
    model.change_data(idn_lb2, idn_ub2, train_t2, train_x2, train_u2,
                      "model/saved.model")
    model.train_u(args.niter, "model/saved.model", args.scipyopt)
    model.train_f(args.niter, "model/saved.model", args.scipyopt)

    idn_t_star1 = idn_data1["idn_t_star"]
    idn_x_star1 = idn_data1["idn_x_star"]
    idn_u_star1 = idn_data1["idn_u_star"]

    idn_t_star2 = idn_data2["idn_t_star"]
    idn_x_star2 = idn_data2["idn_x_star"]
    idn_u_star2 = idn_data2["idn_u_star"]

    idn_u_pred, idn_f_pred = model.identifier_predict(idn_t_star1, idn_x_star1)
    idn_error_u = np.linalg.norm(idn_u_star1 - idn_u_pred, 2) / np.linalg.norm(
        idn_u_star1, 2)
    model.logger.info(f"Data1 Error u: {idn_error_u:.3e}")

    idn_u_pred, idn_f_pred = model.identifier_predict(idn_t_star2, idn_x_star2)
    idn_error_u = np.linalg.norm(idn_u_star2 - idn_u_pred, 2) / np.linalg.norm(
        idn_u_star2, 2)
    model.logger.info(f"Data2 Error u: {idn_error_u:.3e}")

    sol_t_star = sol_data["sol_t_star"]
    sol_x_star = sol_data["sol_x_star"]
    sol_u_star = sol_data["sol_u_star"]

    model.train_solver(args.niter * 2, args.scipyopt)
    u_pred, f_pred = model.solver_predict(sol_t_star, sol_x_star)
    error_u = np.linalg.norm(sol_u_star - u_pred, 2) / np.linalg.norm(
        sol_u_star, 2)
    model.logger.info(f"Error u: {error_u:.3e}")

    sol_X_star = sol_data["sol_X_star"]
    sol_T = sol_data["sol_T"]
    sol_X = sol_data["sol_X"]
    sol_exact = sol_data["sol_exacts"]

    U_pred = griddata(
        sol_X_star, u_pred.flatten(), (sol_T, sol_X), method="cubic")
    plt_saver(U_pred, sol_exact, sol_lb, sol_ub, "Burgers_different")
