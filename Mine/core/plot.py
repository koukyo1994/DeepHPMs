import sys
import matplotlib
matplotlib.use("agg")
sys.path.append("..")


def plt_saver(u_pred, sol_exact, sol_lb, sol_ub):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    from Codes.plotting import newfig, savefig

    fig, ax = newfig(1.0, 0.6)
    ax.axis("off")

    gs = gridspec.GridSpec(1, 2)
    gs.update(top=0.8, bottom=0.2, left=0.2, right=0.9, wspace=0.5)
    ax = plt.subplot(gs[:, 0])
    h = ax.imshow(
        sol_exact,
        interpolation="nearest",
        cmap="jet",
        extent=[sol_lb[0], sol_ub[0], sol_lb[1], sol_ub[1]],
        origin="lower",
        aspect="auto")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h, cax=cax)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    ax.set_title("Exact Dynamics", fontsize=10)

    ax = plt.subplot(gs[:, 1])
    h = ax.imshow(
        u_pred,
        interpolation="nearest",
        cmap="jet",
        extent=[sol_lb[0], sol_ub[0], sol_lb[1], sol_ub[1]],
        origin="lower",
        aspect="auto")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h, cax=cax)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.set_title('Learned Dynamics', fontsize=10)
    savefig("figures/Burgers_Extrapolate")
