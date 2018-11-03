import sys
import numpy as np
import scipy.io
import argparse
from pyDOE import lhs
from scipy.interpolate import griddata
from core.plot import plt_saver

if __name__ == "__main__":
    sys.path.append("./")
    from models.burgers import BurgersHPM

    parser = argparse.ArgumentParser()
    parser.add_argument("--niter", default=10000, type=int)
    parser.add_argument("--scipyopt", default=False)
    args = parser.parse_args()
