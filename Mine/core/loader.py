import numpy as np
import scipy.io
from pyDOE import lhs


class DataLoader:
    def __init__(self, idn_path, sol_path, x_max, y_max):
        self.path_dict = {p: i for i, p in enumerate(idn_path)}

        self.sol_lb = np.array([0.0, -y_max])
        self.sol_ub = np.array([x_max, y_max])

        self.idn_lb = np.array([0.0, -y_max])
        self.idn_ub = np.array([x_max, y_max])

        self.idn_lbs = [np.array([0.0, -y_max]) for _ in idn_path]
        self.idn_ubs = [np.array([x_max, y_max]) for _ in idn_path]

        self.idn_datas = [scipy.io.loadmat(p) for p in idn_path]

        self.idn_ts = [d["t"].flatten()[:, None] for d in self.idn_datas]
        self.idn_xs = [d["x"].flatten()[:, None] for d in self.idn_datas]
        self.idn_exacts = [np.real(d["usol"]) for d in self.idn_datas]

        self.idn_Ts = list()
        self.idn_Xs = list()
        for t, x in zip(self.idn_ts, self.idn_xs):
            idn_T, idn_X = np.meshgrid(t, x)
            self.idn_Ts.append(idn_T)
            self.idn_Xs.append(idn_X)

        self.sol_data = scipy.io.loadmat(sol_path)
        self.sol_t = self.sol_data["t"].flatten()[:, None]
        self.sol_x = self.sol_data["x"].flatten()[:, None]
        self.sol_exact = np.real(self.sol_data["usol"])

    def get_solver_data(self, N_f):
        N0 = self.sol_exact.shape[0]
        N_b = self.sol_exact.shape[1]

        idx_x = np.random.choice(self.sol_x.shape[0], N0, replace=False)
        train_x0 = self.sol_x[idx_x, :]
        train_u0 = self.sol_exact[idx_x, 0:1]

        idx_t = np.random.choice(self.sol_t.shape[0], N_b, replace=False)
        train_tb = self.sol_t[idx_t, :]
        train_X_f = self.sol_lb + (self.sol_ub - self.sol_lb) * lhs(2, N_f)

        sol_T, sol_X = np.meshgrid(self.sol_t, self.sol_x)
        sol_t_star = sol_T.flatten()[:, None]
        sol_x_star = sol_X.flatten()[:, None]
        sol_X_star = np.hstack([sol_t_star, sol_x_star])
        sol_u_star = self.sol_exact.flatten()[:, None]

        return {
            "train_x0": train_x0,
            "train_u0": train_u0,
            "train_tb": train_tb,
            "train_X_f": train_X_f,
            "sol_T": sol_T,
            "sol_X": sol_X,
            "sol_x_star": sol_x_star,
            "sol_t_star": sol_t_star,
            "sol_u_star": sol_u_star,
            "sol_X_star": sol_X_star,
            "sol_lb": self.sol_lb,
            "sol_ub": self.sol_ub,
            "sol_exact": self.sol_exact
        }

    def get_train_data(self, path, keep=2 / 3, N_train=10000, noise=0.0):
        path_idx = self.path_dict[path]
        index = int(keep * self.idn_ts[path_idx].shape[0])

        idn_T = self.idn_Ts[path_idx][:, 0:index]
        idn_X = self.idn_Xs[path_idx][:, 0:index]
        idn_exact = self.idn_exacts[path_idx][:, 0:index]

        idn_t_star = idn_T.flatten()[:, None]
        idn_x_star = idn_X.flatten()[:, None]
        idn_u_star = idn_exact.flatten()[:, None]

        idx = np.random.choice(idn_t_star.shape[0], N_train, replace=False)
        train_t = idn_t_star[idx, :]
        train_x = idn_x_star[idx, :]
        train_u = idn_u_star[idx, :]

        train_u = train_u + noise * np.std(train_u) * np.random.randn(
            train_u.shape[0], train_u.shape[1])
        return {
            "train_t": train_t,
            "train_x": train_x,
            "train_u": train_u,
            "idn_t_star": idn_t_star,
            "idn_x_star": idn_x_star,
            "idn_u_star": idn_u_star,
            "idn_lb": self.idn_lb,
            "idn_ub": self.idn_ub
        }

    def get_train_batch(self, keep=2 / 3, N_train=10000, noise=0.0):
        indexes = [int(keep * t.shape[0]) for t in self.idn_ts]
        idn_Ts = [T[:, 0:i] for T, i in zip(self.idn_Ts, indexes)]
        idn_Xs = [X[:, 0:i] for X, i in zip(self.idn_Xs, indexes)]
        idn_exatcs = [E[:, 0:i] for E, i in zip(self.idn_exacts, indexes)]

        idn_t_stars = [T.flatten()[:, None] for T in idn_Ts]
        idn_x_stars = [X.flatten()[:, None] for X in idn_Xs]
        idn_u_stars = [U.flatten()[:, None] for U in idn_exatcs]

        idxs = [
            np.random.choice(idn_t_stars[0].shape[0], N_train, replace=False)
            for _ in idn_t_stars
        ]
        train_ts = [t[i, :] for t, i in zip(idn_t_stars, idxs)]
        train_xs = [x[i, :] for x, i in zip(idn_x_stars, idxs)]
        train_us = [u[i, :] for u, i in zip(idn_u_stars, idxs)]

        train_us = [
            u + noise * np.std(u) * np.random.randn(u.shape[0], u.shape[1])
            for u in train_us
        ]
        return {
            "train_ts": train_ts,
            "train_xs": train_xs,
            "train_us": train_us,
            "idn_t_stars": idn_t_stars,
            "idn_u_stars": idn_u_stars,
            "idn_x_stars": idn_x_stars,
            "idn_lbs": self.idn_lbs,
            "idn_ubs": self.idn_ubs
        }
