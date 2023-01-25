''' Utilities for Neural Networks in PyTorch. '''

import collections as coll
import os
import random
from platform import processor

import numpy as np
import torch as th

from data_utils import batch_iterator

####################################################################################################

def seed_everything(seed=1234):
    # https://github.com/pytorch/pytorch/issues/11278
    random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.backends.cudnn.deterministic = True
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

####################################################################################################

def get_device(*, cuda=True):
    ''' Returns GPU (cuda=True) or CPU device (cuda=False). '''

    if cuda and th.cuda.is_available():
        device = th.device("cuda:0")
        dname = th.cuda.get_device_name(device)
    else:
        device = th.device("cpu")
        dname = processor()

    print(f'Device ID is "{str(device)}" with full name: {dname}')

    return device

####################################################################################################

def count_parameters(model, text_io=None):
    total_param = 0

    for module in model.children():
        # https://stackoverflow.com/a/48395991/10546849
        for param in module.parameters():
            if param.requires_grad:
                total_param += np.prod(param.size())

    if text_io:
        print(model, "\nTrainable parameters:", total_param, file=text_io)

    return total_param

####################################################################################################

def calculate_loss_error(test_data, model, loss_fn, text_io, stop_after=1024):
    ''' Returns avg loss, error rate, and errors for a given test set. '''

    nbatches, num_all, sum_loss, errors = 0, 0, 0.0, []
    error = coll.namedtuple('Error', 'index image label prediction')

    for inputs, labels in batch_iterator(test_data, stop_after):
        nbatches += 1
        outputs = model(inputs)
        outputs = th.squeeze(outputs)
        sum_loss += loss_fn(outputs, labels).item()

        pred_cat = th.round(outputs) # For binary classification
        # pred_cat = th.argmax(outputs, 1) # For mutiple classes.

        errors += [error(num_all+j, inputs[j], labels[j].item(), pred_cat[j].item())
                   for j in np.where((labels != pred_cat).cpu().numpy())[0]]
        num_all += len(labels)
        if num_all >= stop_after:
            break

    loss = sum_loss/nbatches # Loss is an average of a batch
    error_rate = len(errors)/num_all

    if text_io:
        print("Average loss: %.3f, Error rate: %.3f " % (loss, error_rate), file=text_io)

    return loss, error_rate, errors

####################################################################################################

def evaluate(data, model, loss_fn, text_io):
    ''' Calls calculate_loss_error() in appropriate context.
    Returns avg loss, error rate, and errors for a given test set. '''
    model.eval()
    with th.no_grad():
        return calculate_loss_error(data, model, loss_fn, text_io=text_io)

####################################################################################################

def optimizer_to_device(optimizer, device):
    ''' https://github.com/pytorch/pytorch/issues/2830#issuecomment-336194949 '''
    for state in optimizer.state.values():
        for key, val in state.items():
            if th.is_tensor(val):
                state[key] = val.to(device)

####################################################################################################
