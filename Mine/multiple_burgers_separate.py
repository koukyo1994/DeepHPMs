import sys
import numpy as np
import argparse

from scipy.interpolate import griddata
from core.plot import plt_saver
from core.loader import DataLoader

from hyperdash import Experiment

if __name__ == "__main__":
    sys.path.append("./")
    from models.burgers_train_separate import BurgersSeparate
    parser = argparse.ArgumentParser()
    parser.add_argument("--niter", default=10000, type=int)
    parser.add_argument("--scipyopt", default=False)
    parser.add_argument("--name", default="default")
    parser.add_argument("--traindata", nargs="+")
    parser.add_argument(
        "--testdata", default="../MyData/burgers_polynominal.mat")
    args = parser.parse_args()
    logname = f"log/burgers_{args.name}.log"
    figurename = f"Burgers_{args.name}"
    # filen = "../MyData/burgers_cos.mat"

    exp = Experiment(args.name)
    exp.param("niter", args.niter)
    exp.param("scipyopt", args.scipyopt)
    exp.param("testdata", args.testdata)
    for i, n in enumerate(args.traindata):
        exp.param(f"traindata{i}", n)

    dataloader = DataLoader(args.traindata, args.testdata, 10.0, 8.0)
    sol_data = dataloader.get_solver_data(20000)
    idn_data = dataloader.get_train_batch()

    u_layers = [[2, 50, 50, 50, 50, 1] for _ in range(len(args.traindata))]
    pde_layers = [3, 100, 100, 1]
    layers = [2, 50, 50, 50, 50, 1]

    idn_lbs = idn_data["idn_lbs"]
    idn_ubs = idn_data["idn_ubs"]
    train_ts = idn_data["train_ts"]
    train_xs = idn_data["train_xs"]
    train_us = idn_data["train_us"]

    train_x0 = sol_data["train_x0"]
    train_u0 = sol_data["train_u0"]
    sol_lb = sol_data["sol_lb"]
    sol_ub = sol_data["sol_ub"]
    train_tb = sol_data["train_tb"]
    train_X_f = sol_data["train_X_f"]

    model = BurgersSeparate(idn_lbs, idn_ubs, sol_lb, sol_ub, train_ts,
                            train_xs, train_us, train_tb, train_x0, train_u0,
                            train_X_f, layers, u_layers, pde_layers, logname)
    model.train_idn(args.niter, "model/saved.model", args.scipyopt)

    # idn_t_stars = idn_data["idn_t_stars"]
    # idn_x_stars = idn_data["idn_x_stars"]
    # idn_u_stars = idn_data["idn_u_stars"]

    # idn_u_preds, idn_f_preds = model.idn_predict(idn_t_stars, idn_x_stars)
    # idn_error_us = [
    #     np.linalg.norm(star - pred, 2) / np.linalg.norm(star, 2)
    #     for star, pred in zip(idn_u_stars, idn_u_preds)
    # ]
    # for i, e in enumerate(idn_error_us):
    #     model.logger.info(f"Data{i} Error u: {e:.3e}")

    sol_t_star = sol_data["sol_t_star"]
    sol_x_star = sol_data["sol_x_star"]
    sol_u_star = sol_data["sol_u_star"]

    model.train_solver(args.niter * 2, args.scipyopt)
    u_pred, f_pred = model.solver_predict(sol_t_star, sol_x_star)
    error_u = np.linalg.norm(sol_u_star - u_pred, 2) / np.linalg.norm(
        sol_u_star, 2)
    model.logger.info(f"Error u: {error_u:.3e}")
    exp.metric("Error u", error_u)

    sol_X_star = sol_data["sol_X_star"]
    sol_T = sol_data["sol_T"]
    sol_X = sol_data["sol_X"]
    sol_exact = sol_data["sol_exact"]
    U_pred = griddata(
        sol_X_star, u_pred.flatten(), (sol_T, sol_X), method="cubic")
    plt_saver(U_pred, sol_exact, sol_lb, sol_ub, figurename)
    exp.end()
