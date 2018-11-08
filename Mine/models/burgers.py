import tensorflow as tf

from core.model_base import BaseHPM


class BurgersHPM(BaseHPM):
    def __init__(self, idn_lbs, idn_ubs, sol_lbs, sol_ubs, t, x, u, tb, x0, u0,
                 X_f, layers, u_layers, pde_layers):
        super().__init__(idn_lbs, idn_ubs, sol_lbs, sol_ubs, t, x, u, tb, x0,
                         u0, X_f, layers, u_layers, pde_layers)
        # Initialization for Identification
        self.idn_init(t, x, u, u_layers, pde_layers)

        # Initialization for Solver
        self.sol_init(x0, u0, tb, X_f, layers)

        # TF session
        self.sess = tf.Session(
            config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def identifier_f(self, t, x):
        us = map(self.idn_net, t, x)
        u_ts = map(lambda u, t: tf.gradients(u, t)[0], us, t)
        u_xs = map(lambda u, x: tf.gradients(u, x)[0], us, x)
        u_xxs = map(lambda u, x: tf.gradients(u, x)[0], u_xs, x)

        terms = map(lambda u, u_x, u_xxs: tf.concat([u, u_x, u_xxs], 1), us,
                    u_xs, u_xxs)
        fs = map(lambda u_t, terms: u_t - self.pde_net(terms), u_ts, terms)
        return fs

    def solver_net_f(self, t, x):
        u, _ = self.solver_net_u(t, x)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        terms = tf.concat([u, u_x, u_xx], 1)

        f = u_t - self.pde_net(terms)
        return f
