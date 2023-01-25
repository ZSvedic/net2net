import io
import sys
import time

import numpy as np

import models
import data_utils as data_ut
# import display_utils as disp_ut
import nn_utils as nn_ut
import trainer as tr

####################################################################################################

def experiment_plain_fc_nn(dsets, device):
    nn_ut.seed_everything()
    model = models.ModelNFC(False, True, None, dsets.in_dim, 48, 24, 12, dsets.out_dim)

    # out_params = {'output':tr.Output.none}
    # out_params = {'output':tr.Output.text, 'text_io':sys.stdout}
    # out_params = {'output':tr.Output.plot, 'text_io':io.StringIO()}
    out_params = {'output':tr.Output.png, 'text_io':io.StringIO(), \
        'fname':'plain-FC-'+time.strftime("%H-%M-%S")}

    tr.train_and_report(dsets, device, model, 0.0040, 80, 701, **out_params)

####################################################################################################

def experiment_grid_bnorm(dsets, device, n, lr_range, bsizes_range, save):
    for bn in [False, True]:
        nn_ut.seed_everything()
        model = models.ModelNFC(False, bn, None, dsets.in_dim, 24, 12, dsets.out_dim)
        sio = io.StringIO()
        nn_ut.count_parameters(model, text_io=sio)
        tr.plot_grid_search(dsets, device, model, 101, n, lr_range, bsizes_range, save, \
            output=tr.Output.png, text_io=sio, fname=f'grid({n})-bn({bn})')

####################################################################################################

def experiment_init_bnorm_dropout(dsets, device, save, *, ol, bnl, dl):
    results = np.zeros((len(ol), len(bnl), len(dl)))
    formatter = 'o({})-bn({})-d({})'.format

    for oi, o in enumerate(ol):
        for bni, bn in enumerate(bnl):
            for di, d in enumerate(dl):
                sio = io.StringIO() if save else sys.stdout
                model = models.ModelNFC(o, bn, d, dsets.in_dim, 48, 24, 12, dsets.out_dim)
                results[oi, bni, di], _ = tr.train_and_report(dsets, device, model, \
                    0.0040, 80, 301, output=tr.Output.png, text_io=sio, fname=formatter(o, bn, d))

    o_win, bn_win, d_win = [0]*len(ol), [0]*len(bnl), [0]*len(dl)

    minv, min_name = sys.maxsize, None
    for oi, o in enumerate(ol):
        for bni, bn in enumerate(bnl):
            for di, d in enumerate(dl):
                current = results[oi, bni, di]
                if current < minv:
                    minv, min_name = current, formatter(o, bn, d)
                o_win[oi] += (min(results[:, bni, di]) == current)
                bn_win[bni] += (min(results[oi, :, di]) == current)
                d_win[di] += (min(results[oi, bni, :]) == current)

    print('o_win:', *zip(ol, o_win))
    print('bn_win:', *zip(bnl, bn_win))
    print('d_win:', *zip(dl, d_win))
    print('Best is %.5f' % minv, 'at', min_name)

####################################################################################################

def main():
    nn_ut.seed_everything() # For determinism, always generate the same random numbers.
    dsets = data_ut.create_spiral_datasets(4*360) # Create datasets
    # disp_ut.plot_nndatasets(dsets) # Sanity check
    device = nn_ut.get_device(cuda=False) # Device

    # Run experiments:

    # experiment_plain_fc_nn(dsets, device)

    # experiment_grid_bnorm(dsets, device, 16, (0.001, 0.02), (20, 100), True)

    # experiment_init_bnorm_dropout(dsets, device, True, \
    #     ol=[False, True], bnl=[False, True], dl=[None, 0.2])
    experiment_init_bnorm_dropout(dsets, device, True, ol=[False], bnl=[True], dl=[None, 0.2])
####################################################################################################

if __name__ == '__main__':
    main()

####################################################################################################
