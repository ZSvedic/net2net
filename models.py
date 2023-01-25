from random import sample
import torch as th
import torch.nn as nn

####################################################################################################

class Block(nn.Module):
    ''' One NN building block: FC, activation, batch norm (optional) and dropout (optional). '''

    def __init__(self, in_size, out_size, orth_init, activation, batch_norm, dropout):
        super().__init__()

        fc = nn.Linear(in_size, out_size)
        fc.name = 'FC({}x{})'.format(in_size, out_size)
        if orth_init:
            th.nn.init.orthogonal_(fc.weight)
            fc.bias.data.fill_(0.01)

        activation.name = activation.__class__.__name__

        l = [fc, activation]

        if batch_norm:
            bn = nn.BatchNorm1d(out_size)
            bn.name = 'BN({})'.format(out_size)
            l.append(bn)

        if dropout:
            d = nn.Dropout(p=dropout)
            d.name = 'D({})'.format(dropout)
            l.append(d)

        self.layers = nn.ModuleList(l)

    def forward(self, *x):
        x = x[0] # Function has only one input argument
        for layer in self.layers:
            x = layer(x)
        return x

    def __repr__(self):
        return ' '.join([l.name for l in self.layers])

####################################################################################################

class ModelNFC(nn.Module):
    ''' Fully-connected model with N hidden layers. '''

    def __init__(self, orth_init, batch_norm, dropout, *sizes):
        super().__init__()

        l = []
        for i in range(len(sizes)-2):
            l.append(Block(sizes[i], sizes[i+1], orth_init, nn.ReLU(), batch_norm, dropout))
        l.append(Block(sizes[-2], sizes[-1], orth_init, nn.Sigmoid(), False, None))

        self.blocks = nn.ModuleList(l)

    def forward(self, *x):
        x = x[0] # Function has only one input argument
        for block in self.blocks:
            x = block(x)
        return x

    def __repr__(self):
        return '\n'.join([str(b) for b in self.blocks])

####################################################################################################

def delete_adam_params(optim, params):
    for gr in optim.param_groups:
        opt_params = gr['params']
        for i, t in reversed(list(enumerate(opt_params))):
            for j, p in enumerate(params):
                # pylint: disable=W0212
                if p._cdata == t._cdata:
                    del opt_params[i]
                    del params[j]
                    if not params:
                        return
                    break

    assert not params, "Some parameters were not found in the optimizer!"

####################################################################################################

def add_neurons(old_data, add_indices, scaling, dim):
    eps = 2e-3
    if dim == 0:
        new_neur = (old_data[add_indices]+eps)/scaling
    else: # dim = 1
        new_neur = (old_data[:, add_indices]+eps)/scaling
    new_data = th.cat(((old_data-eps)/scaling, new_neur), dim=dim)
    return new_data

####################################################################################################

def widen_layer(layers, i_layer0, i_layer1, optim, device, scaling, random):
    # Get old expanded (0) and the next (1) layer:
    old0 = layers[i_layer0]
    old1 = layers[i_layer1]

    # Initalize new, larger layers:
    old_size = old0.out_features
    new_n_neur = int(old_size*scaling)
    to_add = new_n_neur-old_size
    scaling = new_n_neur/old_size # Adjust scaling for integer rounding.
    new0 = nn.Linear(old0.in_features, new_n_neur)
    new0.name = "Widen" + str(new_n_neur)
    new1 = nn.Linear(new_n_neur, old1.out_features)
    new1.name = "AfterWiden"

    # Copy weights (adjusted for epsilon), biases, divide by two and copy to device:
    if random:
        add_indices = sample(list(range(old_size)), to_add)
        #(np.random.rand(to_add) * old_size).astype(int)
    else:
        add_indices = list(range(to_add))
    new0.weight.data = add_neurons(old0.weight.data, add_indices, scaling, dim=0)
    new0.bias.data = add_neurons(old0.bias.data, add_indices, scaling, dim=0)
    new1.weight.data = add_neurons(old1.weight.data, add_indices, scaling, dim=1)
    new1.bias.data = old1.bias.data

    # Copy to device:
    new0.to(device)
    new1.to(device)

    # Delete old params
    delete_adam_params(optim, [old0.weight.data, old0.bias.data, old1.weight.data, old1.bias.data])

    # Flatten list of params: https://stackoverflow.com/a/952952/10546849
    new_params = [p for l in (new0, new1) for p in l.parameters()]

    # Add new params to optimizer: http://tinyurl.com/y5625rby
    optim.add_param_group({'params': new_params})

    # Use new layers:
    layers[i_layer0] = new0
    layers[i_layer1] = new1

    # Batch norm resizing:
    if i_layer1-i_layer0 == 2: # Was there batch norm between FC layers?
        layers[i_layer0+1] = nn.BatchNorm1d(new_n_neur)
        layers[i_layer0+1].name = "BN_Widen"

####################################################################################################

def insert_identity(layers, n_neur, i_layer, batch_norm, optim, device):
    if batch_norm:
        ins_batch = nn.BatchNorm1d(n_neur)
        ins_batch.name = "BN_Identity"
        ins_batch.to(device)
        layers.insert(i_layer, ins_batch)

    ins_lin = nn.Linear(n_neur, n_neur)
    ins_lin.name = "Identity"
    th.eye(n_neur, out=ins_lin.weight.data, requires_grad=True)
    ins_lin.bias.data = th.zeros_like(ins_lin.bias.data)
    ins_lin.to(device)
    layers.insert(i_layer, ins_lin)

    # https://discuss.pytorch.org/t/how-to-change-the-adam-optimizers-parameters-when-adding-neurons/44989
    for p in ins_lin.parameters():
        optim.add_param_group({'params': [p]})

####################################################################################################

def expander(layers, optim, device):
    i_lin = [i for i, l in enumerate(layers) if isinstance(l, nn.Linear)]

    # Insert identity layer at the end
    batch_norm = (i_lin[-1] - i_lin[-2]) == 2
    insert_identity(layers, layers[i_lin[-2]].out_features, i_lin[-1], batch_norm, optim, device)

    print("Layer names:", *[l.name for l in layers])

    # NN should have a shape of triangle, with every next layer being smaller for scaling factor.
    # Find a first layer that deverges from that shape.
    scaling = 1.5
    for j in range(len(i_lin)-1):
        if layers[i_lin[j]].out_features < int(layers[i_lin[j+1]].out_features*scaling):
            widen_layer(layers, i_lin[j], i_lin[j+1], optim, device, scaling, True)

####################################################################################################
