import os.path as osp

from tensorboardX import SummaryWriter

####################################################################################################

class TBWriters:
    ''' Helper class for TensorBoradX writers. '''

    def __init__(self, path):
        if not osp.isdir(path):
            raise FileNotFoundError("Argument '"+path+"' is not a valid directory.")
        self.train = SummaryWriter(osp.join(path, "train"))
        self.validate = SummaryWriter(osp.join(path, "validate"))
        self.test = SummaryWriter(osp.join(path, "test"))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.train.close()
        self.validate.close()
        self.test.close()
        return False

    def append(self, epoch, train_loss, validate_loss, error_rate, named_parameters):
        ''' Saves losses and model params to TensorBoradX log file. '''

        self.train.add_scalar('data/loss', train_loss, epoch)
        self.validate.add_scalar('data/loss', validate_loss, epoch)
        self.test.add_scalar('data/error_rate', error_rate, epoch)

        for name, param in named_parameters:
            self.train.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

####################################################################################################
