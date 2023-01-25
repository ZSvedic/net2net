''' Classes and helper methods needed for training. '''

import pickle
import sys
from enum import Enum
from math import isclose
from unittest.mock import MagicMock

import matplotlib.pyplot as plt
import torch as th

import data_utils as data_ut
import display_utils as disp_ut
import nn_utils as nn_ut
from execution_timer import ExecutionTimer
from io_dict_file import IODictFile
from nn_training_log import NNTrainingLog
from tb_writers import TBWriters

####################################################################################################

class Trainer:
    ''' Class that holds together variables needed for NN training.
    __init__ only takes named parameters. '''

    def __init__(self, *, data, model, optimizer, device, loss_fn, \
        batch_size=16, logger=MagicMock(), path=None, print_period=50, expander=None, \
        text_io=None, axes=disp_ut.PlotAxes(None, None, None, None)):

        # Required:
        self.data = data
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = loss_fn

        # Optional:
        self.batch_size = batch_size
        self.logger = logger
        self.path = path
        self.print_period = print_period
        self.expander = expander
        self.text_io = text_io
        self.axes = axes

        # Internal:
        self.epoch = 0
        self.expansions = []

        if path is not None:
            self.tb_writer = TBWriters(path)
            self.io = IODictFile(path, "model.chk", self.get_dict, self.set_dict)
            self.io.load(file_required=False)
        else:
            self.tb_writer = MagicMock()
            self.io = MagicMock()

        self.to_device()

    def to_device(self):
        self.model.to(self.device)
        self.loss_fn.to(self.device)

        for ds in (self.data.train, self.data.validate, self.data.test):
            if issubclass(type(ds.inputs), th.Tensor):
                break # If inputs are already Tensors, no need to move anything.
            ds.inputs = th.from_numpy(ds.inputs).float().to(self.device)
            ds.labels = th.from_numpy(ds.labels).float().to(self.device)

        nn_ut.optimizer_to_device(self.optimizer, self.device)

    def get_dict(self):
        return {
            'model_dict': self.model.state_dict(),
            'optimizer_dict': self.optimizer.state_dict(),
            'logger': self.logger,
            'epoch': self.epoch}

    def set_dict(self, d):
        self.model.load_state_dict(d['model_dict'])
        self.optimizer.load_state_dict(d['optimizer_dict'])
        self.logger = d['logger']
        self.epoch = d['epoch']

    def is_print_epoch(self):
        return self.text_io and self.epoch%self.print_period == 0

    def validate(self, train_loss):
        # Evaluate epoch

        validate_loss, validate_error_rate, _ = nn_ut.evaluate(
            self.data.validate, self.model, self.loss_fn, \
                sys.stdout if self.is_print_epoch() else None)

        # Custom logger
        is_best = self.logger.append(
            train_loss, validate_loss, validate_error_rate)

        # TensorBoardX logging
        self.tb_writer.append(self.epoch, train_loss, validate_loss, \
            validate_error_rate, self.model.named_parameters())

        # Save checkpoint
        self.io.save(is_best)

        # Plot decision boundary and training charts
        if self.is_print_epoch():
            if disp_ut.plot_decision_boundary(self.axes.bound, self.axes.colb, \
                self.data.train.inputs, self.data.train.labels, self.model, \
                self.device, self.epoch) and disp_ut.plot_loss_error(self.logger, self.axes.log):
                plt.pause(0.01)

        return validate_loss, validate_error_rate

    def expand_if_needed(self, validate_loss):
        if self.expander is not None and \
            self.print_epoch() and self.epoch >= self.print_period:

            change, path = self.logger.loss_relative_change_and_path(self.print_period)

            if change < 0.05 and path < 0.25: # Change was small and oscillations were not large
                self.expanding_strategy(self.model.layers, self.optimizer, self.device) # Expand

                nparam = nn_ut.count_parameters(self.model, text_io=sys.stdout) # Print # of params
                self.expansions.append((self.epoch, nparam)) # Log expansion

                if __debug__: # Check that expansion didn't change output
                    new_loss, _, _ = nn_ut.evaluate(self.data.validate, self.model, \
                        self.loss_fn, None)
                    if not isclose(new_loss, validate_loss, rel_tol=0.25):
                        print("Expansion changed model output!")

    def train_one_epoch(self):

        self.epoch += 1
        epoch_loss, nbatches = 0.0, 0

        if self.is_print_epoch():
            print("Epoch", str(self.epoch), end=': ')

        self.model.train()

        for inputs, labels in data_ut.batch_iterator(self.data.train, self.batch_size):

            # Forward prop
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            outputs = th.squeeze(outputs)

            # Backward prop
            loss = self.loss_fn(outputs, labels)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()

            nbatches += 1
            if nbatches%100 == 99:
                print('.', end='')

        # Loss is average of a batch:
        # https://pytorch.org/docs/stable/nn.html#torch.nn.CrossEntropyLoss
        return epoch_loss / nbatches

####################################################################################################

Output = Enum('Output', 'none text plot png')

def check_out_params(output, text_io, fname):
    if __debug__:
        if output == Output.none:
            assert not text_io and not fname
        elif output in (Output.text, Output.plot):
            assert text_io and not fname
        else: # Output.png
            assert text_io and fname

####################################################################################################

def train_and_report(dsets, device, model, lr, bsize, epochs, *, \
    expander=None, output=Output.none, text_io=None, fname=None):

    check_out_params(output, text_io, fname)

    # Create optimizer
    optim = th.optim.Adam(model.parameters(), lr=lr)

    # Plot stuff.
    if output.value >= Output.plot.value:
        print(f'fname:{fname} Adam_lr:{lr} bsize:{bsize} epochs:{epochs}', file=text_io)
        axes = disp_ut.create_reporting_plots()
    else:
        axes = disp_ut.PlotAxes(None, None, None, None)

    # Text output stuff.
    console = (output != Output.none)
    if console:
        nn_ut.count_parameters(model, text_io=text_io)
        disp_ut.print_dashes("Training:")

    # Create Trainer class with everything related to training.
    trainer = Trainer(
        data=dsets, model=model, optimizer=optim, device=device,
        loss_fn=th.nn.BCELoss(), # 2 classes -> BCELoss(), 2+ classes -> CrossEntropyLoss()
        batch_size=bsize,
        logger=NNTrainingLog(),  # NNTrainingLog() or skip argument
        path=None, # None or "RUNS/"
        print_period=100,
        expander=expander, # Expanding requires logger
        text_io=text_io,
        axes=axes)

    # Run training-validate-expand loop.
    with trainer.tb_writer, ExecutionTimer(console) as timer:
        for _ in range(epochs):
            epoch_train_loss = trainer.train_one_epoch() # Train
            epoch_validate_loss, _ = trainer.validate(epoch_train_loss) # Validate
            trainer.expand_if_needed(epoch_validate_loss)  # Expand if needed
    time = timer.duration()

    # Report results.
    if console:
        disp_ut.print_dashes()
        print(timer.duration_str(), file=text_io)
        if trainer.expansions:
            print("Expansions:", *["(Index:"+str(i)+", Params:"+str(nparams)+")" \
            for i, nparams in trainer.expansions], file=text_io)
        trainer.logger.print_best(text_io)
        print("Test set evaluation: ", end="", file=text_io)

    loss = nn_ut.evaluate(dsets.test, trainer.model, trainer.loss_fn, text_io)[0]

    # Plot/png stuff.
    if output.value >= Output.plot.value:
        disp_ut.axes_display_text(trainer.axes.text, text_io.getvalue())
        if output == Output.plot:
            plt.pause(0.01)
            plt.show()
        else:
            plt.savefig('OUTPUT/' + fname + '.png')

    return loss, time

####################################################################################################

def plot_grid_search(dsets, device, model, n_epochs, n_grid, lr_range, batch_range, save, \
    output, text_io, fname):

    assert output.value >= Output.plot.value # This method can only plot and save the plot.
    check_out_params(output, text_io, fname)

    print(f'fname:{fname} n_epochs:{n_epochs} n_grid:{n_grid}', file=text_io)

    lrs = data_ut.get_log_scaled_range(*lr_range, n_grid, random=True)
    batch_sizes = [int(fl) for fl in \
        data_ut.get_log_scaled_range(*batch_range, n_grid, random=True)]
    path = 'OUTPUT/'+fname+'.pickle'

    if save:
        losses_times = [train_and_report(dsets, device, model, lr, bs, n_epochs, \
            output=Output.text, text_io=sys.stdout) for (lr, bs) in zip(lrs, batch_sizes)]
        with open(path, 'wb') as fp:
            pickle.dump(losses_times, fp)
    else:
        with open(path, 'rb') as fp:
            losses_times = pickle.load(fp)

    for i in range(n_grid):
        print('%2d: LR=%.5f batch_size=%2d loss=%.5f time=%.2f' \
            % (i, lrs[i], batch_sizes[i], losses_times[i][0], losses_times[i][1]))

    # Plot/png stuff.
    fig, ax = plt.subplots(1, 2, figsize=[12, 6], dpi=200, gridspec_kw={'width_ratios': [5, 8]})
    fig.subplots_adjust(left=0.04, top=0.9, right=0.9, bottom=0.15, wspace=0.3, hspace=0.2)
    fig.canvas.set_window_title(fname)
    disp_ut.position_current_window()
    disp_ut.axes_display_text(ax[0], text_io.getvalue())
    disp_ut.plot_grid_search(ax[1], \
        'Learning rate', lrs, lr_range, \
        'Batch size', batch_sizes, batch_range, \
        'Test loss', [ls[0] for ls in losses_times])

    if output == Output.plot:
        plt.pause(0.01)
        plt.show()
    else:
        plt.savefig('OUTPUT/' + fname + '.png')

####################################################################################################
