''' Utils for visualizing data. '''

from collections import namedtuple

import numpy as np
import torch as th
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.colors import LogNorm

# IMPORTANT: https://pbpython.com/effective-matplotlib.html

####################################################################################################

def plot_loss_error(logger, ax):
    ''' Plots loss and error rate chart. '''

    if ax is None:
        return False

    epochs = list(range(1, len(logger.train_loss_log)+1))

    ax.clear()

    ax.plot(epochs, logger.train_loss_log, "r-")
    ax.plot(epochs, logger.validate_loss_log, "b-")
    ax.plot(epochs, logger.error_rate_log, "g-")
    ax.legend(['Training Loss', 'Validation Loss', 'Test Error Rate'])

    ax.set(xlabel='Epoch', ylabel='Loss')

    return True

####################################################################################################

def plot_decision_boundary(ax, colb_ax, X, Y, model, device, epoch):
    if ax is None:
        return False

    # https://stackoverflow.com/questions/19054923/plot-decision-boundary-matplotlib

    # X - some data in 2dimensional np.array.
    x_min, x_max = X[:, 0].min()-3, X[:, 0].max()+3
    y_min, y_max = X[:, 1].min()-3, X[:, 1].max()+3
    h = 0.5
    xx, yy = th.meshgrid(th.arange(x_min, x_max, h), th.arange(y_min, y_max, h), indexing=None)

    # "model" is model's prediction (classification) function.
    model.eval()
    inp = np.c_[xx.contiguous().view(-1), yy.contiguous().view(-1)]
    inp = th.from_numpy(inp).float().to(device)
    Z = model(inp)
    Z = Z.reshape(xx.shape).detach().cpu()

    # Clear Axes
    ax.clear()
    colb_ax.clear()

    # https://matplotlib.org/gallery/color/colormap_reference.html
    ma = ax.contourf(xx, yy, Z, [0, 0.17, 0.34, 0.49, 0.51, 0.66, 0.83, 1], cmap=plt.cm.coolwarm)
    # Plot the training points.
    ax.scatter(X[:, 0].cpu(), X[:, 1].cpu(), s=4, c=Y.cpu(), cmap=plt.cm.seismic)
    # Title and colorbar.
    ax.set(title=str(epoch))
    ax.figure.colorbar(ma, cax=colb_ax)

    return True

####################################################################################################

def plot_grid_search(ax, x_name, X, x_range, y_name, Y, y_range, z_name, Z):
    # https://stackoverflow.com/a/25421861/10546849
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    # https://stackoverflow.com/a/1183415/10546849
    ax.set_yscale('log')
    ax.set_xscale('log')
    # https://matplotlib.org/gallery/color/colormap_reference.html
    # https://stackoverflow.com/a/5748203/10546849
    # https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/contourf_demo.html#sphx-glr-gallery-images-contours-and-fields-contourf-demo-py
    levels = [0.5**i for i in reversed(range(8))]
    levels.insert(0, 0.0)
    norm = LogNorm(vmin=levels[1], vmax=1.0)
    ax.set_facecolor('#DCDCDC')
    cmap = get_cmap('Blues_r')
    cmap.set_bad('#042144')
    ma = ax.tricontourf(X, Y, Z, cmap=cmap, levels=levels, norm=norm)
    cb = ax.figure.colorbar(ma, ax=ax, format='%.4f')
    cb.ax.set_ylim([0.0, 1.0])
    # Also plot points
    ax.scatter(X, Y, s=2, c="#B200FF")
    # Chart and axes titles
    ax.set(title=str(len(X))+' points for '+z_name)
    ax.set(xlabel=x_name, ylabel=y_name)

####################################################################################################

def position_current_window(x=400, y=80):
    # https://stackoverflow.com/questions/7449585/how-do-you-set-the-absolute-position-of-figure-windows-with-matplotlib
    mngr = plt.get_current_fig_manager()
    # https://stackoverflow.com/questions/20025077/how-do-i-display-a-matplotlib-figure-window-on-top-of-all-other-windows-in-spyde
    # geom = mngr.window.geometry()
    # _, _, dx, dy = geom.getRect()
    # mngr.window.setGeometry(x, y, dx, dy)

####################################################################################################

def print_dashes(message=""):
    print("-----------------------------------------------------------", message)

####################################################################################################

def axes_display_text(ax, text):
    ax.axis('off')
    txt = ax.text(0, 0, text, wrap=True)
    clip = ax.get_window_extent()
    # pylint: disable=W0212
    txt._get_wrap_line_width = lambda: clip.width

####################################################################################################

PlotAxes = namedtuple("PlotAxes", "text log bound colb")

####################################################################################################

def create_reporting_plots():
    ''' Create plots for progress log, decision boundary, and color bar. '''
    fig, ax = plt.subplots(1, 4, figsize=[13, 4], dpi=200, \
        gridspec_kw={'width_ratios': [20, 20, 20, 1]})
    fig.subplots_adjust(left=0.04, top=0.9, right=0.9, bottom=0.15, wspace=0.3, hspace=0.2)
    axes = PlotAxes(*ax)
    position_current_window()
    return axes

####################################################################################################

def plot_nndatasets(nn_ds):
    ''' Plots each data set on a separate chart. '''

    _, ax = plt.subplots(1, 3, sharex=True, sharey=True, dpi=200, figsize=[13, 4])

    for i, s in enumerate((nn_ds.train, nn_ds.test, nn_ds.validate)):
        ax[i].set_title(s.name)
        for j in range(len(nn_ds.class_names)):
            is_class = (s.labels == j)
            ax[i].plot(s.inputs[is_class, 0], s.inputs[is_class, 1], '.')

    plt.show()

####################################################################################################
