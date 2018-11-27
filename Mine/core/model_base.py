import os
import time
import numpy as np
import tensorflow as tf
import core.nn as nn

from core.log import get_logger
from config.constants import LOG_PATH, ACTIVATION, INTERVAL


class BaseHPM:
    def __init__(self,
                 idn_lbs,
                 idn_ubs,
                 sol_lbs,
                 sol_ubs,
                 t,
                 x,
                 u,
                 tb,
                 x0,
                 u0,
                 X_f,
                 layers,
                 u_layers,
                 pde_layers,
                 log_path=LOG_PATH):

        if not ((len(idn_lbs) == len(idn_ubs)) and (len(idn_ubs) == len(t)) and
                (len(t) == len(u)) and (len(u) == len(x)) and
                (len(x) == len(u_layers)) and (len(tb) == len(x0)) and
                (len(x0) == len(u0)) and (len(u0) == len(X_f)) and
                (len(X_f) == len(layers))):
            assert IndexError("array size mismatch")

        # Boundary Conditions
        self.idn_lbs = idn_lbs
        self.idn_ubs = idn_ubs

        self.sol_lb = sol_lbs
        self.sol_ub = sol_ubs

        self.loss_before = 1e5

        # Logging Tool
        self.logger = get_logger(log_path)

    def idn_init(self, t, x, u, u_layers, pde_layers):
        # Training Data
        self.t = t
        self.x = x
        self.u = u

        # Layers
        self.u_layers = u_layers
        self.pde_layers = pde_layers

        # Weights and Biases
        self.u_params = list(map(nn.initialize_nn, self.u_layers))
        self.pde_weights, self.pde_biases = nn.initialize_nn(self.pde_layers)

        # TF placeholders
        self.t_phs = [tf.placeholder(tf.float32, [None, 1])] * len(u_layers)
        self.u_phs = [tf.placeholder(tf.float32, [None, 1])] * len(u_layers)
        self.x_phs = [tf.placeholder(tf.float32, [None, 1])] * len(u_layers)
        self.terms_phs = tf.placeholder(tf.float32, [None, pde_layers[0]])

        # TF graphs
        self.u_preds = self.idn_net(self.t_phs, self.x_phs)
        self.pde_pred = self.pde_net(self.terms_phs)
        self.f_pred = self.identifier_f(self.t_phs, self.x_phs)
        # Loss
        self.loss = tf.reduce_sum(
            sum(
                list(
                    map(tf.square,
                        map(lambda x, y: x - y, self.u_preds, self.u_phs))) +
                list(map(tf.square, self.f_pred))))
        # Scipy Optimizer
        unnested = []
        for n in self.u_params:
            for p in n:
                unnested += p
        self.scipy_optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            self.loss,
            var_list=unnested + self.pde_weights + self.pde_biases,
            method="L-BFGS-B",
            options={
                "maxiter": 100000,
                "maxfun": 100000,
                "maxcor": 50,
                "maxls": 50,
                "ftol": 1.0 * np.finfo(float).eps
            })

        # Adam Optimizer
        self.adam_optimizer = tf.train.AdamOptimizer()
        self.adam_optimizer_train = self.adam_optimizer.minimize(
            self.loss, var_list=unnested + self.pde_weights + self.pde_biases)

    def idn_net(self, t, x):
        def init(X, idn_lb, idn_ub):
            return 2. * (X - idn_lb) / (idn_ub - idn_lb) - 1.

        def build_nn(H, params, activation):
            return nn.neural_net(H, params[0], params[1], activation)

        X = map(lambda a, b: tf.concat([a, b], 1), t, x)
        H = map(init, X, self.idn_lbs, self.idn_ubs)
        u = list(
            map(build_nn, H, self.u_params, [ACTIVATION] * len(self.u_params)))
        return u

    def pde_net(self, terms):
        pde = nn.neural_net(terms, self.pde_weights, self.pde_biases,
                            ACTIVATION)
        return pde

    def identifier_f(self, t, x):
        us = self.idn_net(t, x)
        u_ts = list(map(lambda u, t: tf.gradients(u, t)[0], us, t))
        u_xs = list(map(lambda u, x: tf.gradients(u, x)[0], us, x))
        u_xxs = list(map(lambda u, x: tf.gradients(u, x)[0], list(u_xs), x))
        terms = list(
            map(lambda u, u_x, u_xxs: tf.concat([u, u_x, u_xxs], 1), us, u_xs,
                u_xxs))
        fs = list(
            map(lambda u_t, terms: u_t - self.pde_net(terms), u_ts, terms))
        return fs

    def train_idn(self, N_iter, model_path, scipy_opt=False):
        tf_dict = {k: v for k, v in zip(self.t_phs, self.t)}
        tf_dict_u = {k: v for k, v in zip(self.u_phs, self.u)}
        tf_dict_x = {k: v for k, v in zip(self.x_phs, self.x)}
        tf_dict.update(tf_dict_u)
        tf_dict.update(tf_dict_x)
        start_time = time.time()
        for i in range(N_iter):
            self.sess.run(self.adam_optimizer_train, tf_dict)
            if i % INTERVAL == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                self.logger.info(f"""
                    idn-Adam,
                    It: {i},
                    Loss: {loss_value:.3e},
                    Time: {elapsed:.2f}
                    """)
                if model_path:
                    if os.path.exists(model_path):
                        os.rmdir(model_path)
                    self.saver.save(self.sess, model_path)
                start_time = time.time()
        if scipy_opt:
            self.scipy_optimizer.minimize(
                self.sess,
                feed_dict=tf_dict,
                fetches=[self.loss],
                loss_callback=self.callback)

    def idn_predict(self, t_stars, x_stars):
        tf_dict = {k: v for k, v in zip(self.t_phs, t_stars)}
        tf_dict_x = {k: v for k, v in zip(self.x_phs, x_stars)}
        tf_dict.update(tf_dict_x)
        u_stars = self.sess.run(self.u_preds, tf_dict)
        f_star = self.sess.run(self.f_pred, tf_dict)
        return u_stars, f_star

    def pde_predict(self, terms_star):
        tf_dict = {self.terms_phs: terms_star}
        pde_star = self.sess.run(self.pde_pred, tf_dict)
        return pde_star

    def sol_init(self, x0, u0, tb, X_f, layers):
        # Initialize the Vector
        X0 = np.concatenate((0 * x0, x0), 1)
        X_lb = np.concatenate((tb, 0 * tb + self.sol_lb[1]), 1)
        X_ub = np.concatenate((tb, 0 * tb + self.sol_ub[1]), 1)

        self.X_f = X_f
        self.t0 = X0[:, 0:1]  # Initial Data (time)
        self.x0 = X0[:, 1:2]  # Initial Data (space)
        self.t_lb = X_lb[:, 0:1]  # Lower Boundary Data (time)
        self.t_ub = X_ub[:, 0:1]  # Upper Boundary Data (time)
        self.x_lb = X_lb[:, 1:2]  # Lower Boundary Data (space)
        self.x_ub = X_ub[:, 1:2]  # Upper Boundary Data (space)
        self.t_f = X_f[:, 0:1]  # Collocation Points (time)
        self.x_f = X_f[:, 1:2]  # Collocation Points (space)
        self.u0 = u0  # Boundary Data

        # Layers for Solution
        self.layers = layers

        # Initialize NNs for Solution
        self.weights, self.biases = nn.initialize_nn(layers)

        # TF placeholders for Solution
        self.t0_placeholder = tf.placeholder(tf.float32, [None, 1])
        self.x0_placeholder = tf.placeholder(tf.float32, [None, 1])
        self.u0_placeholder = tf.placeholder(tf.float32, [None, 1])
        self.t_lb_placeholder = tf.placeholder(tf.float32, [None, 1])
        self.x_lb_placeholder = tf.placeholder(tf.float32, [None, 1])
        self.t_ub_placeholder = tf.placeholder(tf.float32, [None, 1])
        self.x_ub_placeholder = tf.placeholder(tf.float32, [None, 1])
        self.t_f_placeholder = tf.placeholder(tf.float32, [None, 1])
        self.x_f_placeholder = tf.placeholder(tf.float32, [None, 1])

        # TF graphs for Solution
        self.u0_pred, _ = self.solver_net_u(self.t0_placeholder,
                                            self.x0_placeholder)
        self.u_lb_pred, self.u_x_lb_pred = self.solver_net_u(
            self.t_lb_placeholder, self.x_lb_placeholder)
        self.u_ub_pred, self.u_x_ub_pred = self.solver_net_u(
            self.t_ub_placeholder, self.x_ub_placeholder)
        self.solver_f_pred = self.solver_net_f(self.t_f_placeholder,
                                               self.x_f_placeholder)

        # Loss for Solution
        self.solver_loss = \
            tf.reduce_sum(tf.square(self.u0_placeholder - self.u0_pred)) + \
            tf.reduce_sum(tf.square(self.u_lb_pred - self.u_ub_pred)) + \
            tf.reduce_sum(tf.square(self.u_x_lb_pred - self.u_x_ub_pred)) + \
            tf.reduce_sum(tf.square(self.solver_f_pred))

        # Scipy Optimizer for Solution
        self.scipy_solver_optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            self.solver_loss,
            var_list=self.weights + self.biases,
            method="L-BFGS-B",
            options={
                "maxiter": 50000,
                "maxfun": 50000,
                "maxcor": 50,
                "maxls": 50,
                "ftol": 1.0 * np.finfo(float).eps
            })

        # Adam Optimizer for Solution
        self.adam_solver_optimizer = tf.train.AdamOptimizer()
        self.sol_train_op_Adam = self.adam_solver_optimizer.minimize(
            self.solver_loss, var_list=self.weights + self.biases)

    def solver_net_u(self, t, x):
        X = tf.concat([t, x], 1)
        H = 2.0 * (X - self.sol_lb) / (self.sol_ub - self.sol_lb) - 1.0
        u = nn.neural_net(H, self.weights, self.biases, ACTIVATION)
        u_x = tf.gradients(u, x)[0]
        return u, u_x

    def solver_net_f(self, t, x):
        u, _ = self.solver_net_u(t, x)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        terms = tf.concat([u, u_x, u_xx], 1)

        f = u_t - self.pde_net(terms)
        return f

    def callback(self, loss):
        if self.loss_before / loss > 5:
            self.logger.info(f"'L-BFGS-B' Optimizer Loss: {loss:.3e}")
            self.loss_before = loss

    def train_solver(self, N_iter, scipy_opt=False):
        tf_dict = {
            self.t0_placeholder: self.t0,
            self.x0_placeholder: self.x0,
            self.u0_placeholder: self.u0,
            self.t_lb_placeholder: self.t_lb,
            self.x_lb_placeholder: self.x_lb,
            self.t_ub_placeholder: self.t_ub,
            self.x_ub_placeholder: self.x_ub,
            self.t_f_placeholder: self.t_f,
            self.x_f_placeholder: self.x_f
        }
        start_time = time.time()
        for i in range(N_iter):
            self.sess.run(self.sol_train_op_Adam, tf_dict)
            if i % INTERVAL == 10:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.solver_loss, tf_dict)
                self.logger.info(f"""
                    solver, It: {i},
                    Loss: {loss_value:.3e},
                    Time: {elapsed:.2f}""")
                start_time = time.time()
        if scipy_opt:
            self.scipy_solver_optimizer.minimize(
                self.sess,
                feed_dict=tf_dict,
                fetches=[self.solver_loss],
                loss_callback=self.callback)

    def solver_predict(self, t_star, x_star):
        u_star = self.sess.run(self.u0_pred, {
            self.t0_placeholder: t_star,
            self.x0_placeholder: x_star
        })
        f_star = self.sess.run(self.solver_f_pred, {
            self.t_f_placeholder: t_star,
            self.x_f_placeholder: x_star
        })
        return u_star, f_star
