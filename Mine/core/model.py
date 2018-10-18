import os
import time
import numpy as np
import tensorflow as tf
import core.nn as nn
from config.constants import ACTIVATION, INTERVAL, LOG_PATH
from core.log import get_logger


class DeepHPM:
    def __init__(self, idn_lb, idn_ub, t, x, u, tb, x0, u0, X_f, layers,
                 sol_lb, sol_ub, u_layers, pde_layers):
        # Identifier Boundary
        self.idn_lb = idn_lb
        self.idn_ub = idn_ub

        # Solver Boundary
        self.sol_lb = sol_lb
        self.sol_ub = sol_ub

        # Initialization for Identification
        self.identifier_init(t, x, u, u_layers, pde_layers)

        # Initialization for Solver
        self.solver_init(x0, u0, tb, X_f, layers)

        # Model saver
        self.saver = tf.train.Saver()

        # Logging Tool
        self.logger = get_logger(LOG_PATH)

        # TF session
        self.sess = tf.Session(
            config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def identifier_init(self, t, x, u, u_layers, pde_layers):
        # Training Data for Identification
        self.t = t
        self.x = x
        self.u = u

        # Layers for Identification
        self.u_layers = u_layers
        self.pde_layers = pde_layers

        # Initialize NNs for Identification
        self.u_weights, self.u_biases = nn.initialize_nn(u_layers)
        self.pde_weights, self.pde_biases = nn.initialize_nn(pde_layers)

        # TF placeholders
        self.t_placeholder = tf.placeholder(tf.float32, [None, 1])
        self.u_placeholder = tf.placeholder(tf.float32, [None, 1])
        self.x_placeholder = tf.placeholder(tf.float32, [None, 1])
        self.terms_placeholder = tf.placeholder(tf.float32,
                                                [None, pde_layers[0]])

        # TF graphs
        self.u_pred = self.identifier_net(self.t_placeholder,
                                          self.x_placeholder)
        self.pde_pred = self.pde_net(self.terms_placeholder)
        self.f_pred = self.identifier_f(self.t_placeholder, self.x_placeholder)

        # Loss
        self.u_loss = tf.reduce_sum(
            tf.square(self.u_pred - self.u_placeholder) + 
            tf.square(self.f_pred)
        )
        self.f_loss = tf.reduce_sum(tf.square(self.f_pred))

        # Scipy Optimizer
        self.scipy_u_optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            self.u_loss,
            var_list=self.u_weights + self.u_biases,
            method="L-BFGS-B",
            options={
                "maxiter": 50000,
                "maxfun": 50000,
                "maxcor": 50,
                "maxls": 50,
                "ftol": 1.0 * np.finfo(float).eps
            })
        self.scipy_f_optimizer = tf.contrib.opt.ScipyOptimizerInterface(
            self.f_loss,
            var_list=self.pde_weights + self.pde_biases,
            method="L-BFGS-B",
            options={
                "maxiter": 50000,
                "maxfun": 50000,
                "maxcor": 50,
                "maxls": 50,
                "ftol": 1.0 * np.finfo(float).eps
            })

        # Adam Optimizer
        self.adam_u_optimizer = tf.train.AdamOptimizer()
        self.adam_f_optimizer = tf.train.AdamOptimizer()
        self.adam_u_optimizer_train = self.adam_u_optimizer.minimize(
            self.u_loss, var_list=self.u_weights + self.u_biases + self.pde_weights + self.pde_biases)
        self.adam_f_optimizer_train = self.adam_f_optimizer.minimize(
            self.f_loss, var_list=self.pde_weights + self.pde_biases)

    def identifier_net(self, t, x):
        X = tf.concat([t, x], 1)
        H = 2. * (X - self.idn_lb) / (self.idn_ub - self.idn_lb) - 1.
        u = nn.neural_net(H, self.u_weights, self.u_biases, ACTIVATION)
        return u

    def pde_net(self, terms):
        pde = nn.neural_net(terms, self.pde_weights, self.pde_biases,
                            ACTIVATION)
        return pde

    def identifier_f(self, t, x):
        u = self.identifier_net(t, x)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]

        terms = tf.concat([u, u_x, u_xx], 1)
        f = u_t - self.pde_net(terms)
        return f

    def train_u(self, N_iter, model_path, scipy_opt=False):
        tf_dict = {
            self.t_placeholder: self.t,
            self.x_placeholder: self.x,
            self.u_placeholder: self.u
        }
        start_time = time.time()
        for i in range(N_iter):
            self.sess.run(self.adam_u_optimizer_train, tf_dict)
            if i % INTERVAL == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.u_loss, tf_dict)
                self.logger.info(
                    f"u, It: {i}, Loss: {loss_value:.3e}, Time: {elapsed:.2f}")
                if model_path:
                    if os.path.exists(model_path):
                        os.rmdir(model_path)
                    self.saver.save(self.sess, model_path)
                start_time = time.time()
        if scipy_opt:
            self.scipy_u_optimizer.minimize(
                self.sess,
                feed_dict=tf_dict,
                fetches=[self.f_loss],
                loss_callback=self.callback)

    def train_f(self, N_iter, model_path, scipy_opt=False):
        tf_dict = {
            self.t_placeholder: self.t,
            self.x_placeholder: self.x,
            self.u_placeholder: self.u
        }
        start_time = time.time()
        for i in range(N_iter):
            self.sess.run(self.adam_f_optimizer_train, tf_dict)
            if i % INTERVAL == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.f_loss, tf_dict)
                self.logger.info(
                    f"f, It: {i}, Loss: {loss_value:.3e}, Time: {elapsed:.2f}")
                if model_path:
                    if os.path.exists(model_path):
                        os.rmdir(model_path)
                    self.saver.save(self.sess, model_path)
                start_time = time.time()
        if scipy_opt:
            self.scipy_f_optimizer.minimize(
                self.sess,
                feed_dict=tf_dict,
                fetches=[self.f_loss],
                loss_callback=self.callback)

    def identifier_predict(self, t_star, x_star):
        tf_dict = {self.t_placeholder: t_star, self.x_placeholder: x_star}
        u_star = self.sess.run(self.u_pred, tf_dict)
        f_star = self.sess.run(self.f_pred, tf_dict)

        return u_star, f_star

    def pde_predict(self, terms_star):
        tf_dict = {self.terms_placeholder: terms_star}
        pde_star = self.sess.run(self.pde_pred, tf_dict)
        return pde_star

    def change_data(self, idn_lb, idn_ub, t, x, u, model_path):
        # Model Restortion
        self.saver.restore(self.sess, model_path)

        # Assign New Boundary
        self.idn_lb = idn_lb
        self.idn_ub = idn_ub

        # Assign New Data
        self.t = t
        self.x = x
        self.u = u

    def solver_init(self, x0, u0, tb, X_f, layers):
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

        # Initialize NNs for SSolution
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
        self.logger.info(f"'L-BFGS-B' Optimizer Loss: {loss:.3e}")

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
                self.logger.info(
                    f"solver, It: {i}, Loss: {loss_value:.3e}, Time: {elapsed:.2f}"
                )
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
