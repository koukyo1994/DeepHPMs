import tensorflow as tf

from core.model_base import BaseHPM


class BurgersHPM(BaseHPM):
    def __init__(self, idn_lbs, idn_ubs, sol_lbs, sol_ubs, t, x, u, tb, x0, u0,
                 X_f, layers, u_layers, pde_layers, log_path):
        super().__init__(
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
            log_path=log_path)
        # Initialization for Identification
        self.idn_init(t, x, u, u_layers, pde_layers)

        # Initialization for Solver
        self.sol_init(x0, u0, tb, X_f, layers)

        # Model saver
        self.saver = tf.train.Saver()

        # TF session
        self.sess = tf.Session(
            config=tf.ConfigProto(
                allow_soft_placement=True, log_device_placement=True))
        init = tf.global_variables_initializer()
        self.sess.run(init)
