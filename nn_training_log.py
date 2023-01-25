####################################################################################################

class NNTrainingLog:
    ''' Class that logs losses and error rates for every epoch. '''

    def __init__(self):
        self.train_loss_log = []
        self.validate_loss_log = []
        self.error_rate_log = []
        self.best_loss = 1000.0
        self.best_index = 0

    def append(self, train_loss, validate_loss, error_rate):
        epoch = len(self.train_loss_log)

        self.train_loss_log.append(train_loss)
        self.validate_loss_log.append(validate_loss)
        self.error_rate_log.append(error_rate)

        if validate_loss < self.best_loss:
            self.best_loss = validate_loss
            self.best_index = epoch
            return True
        else:
            return False

    def loss_relative_change_and_path(self, n_epochs):
        l = self.validate_loss_log[-n_epochs:]
        change = abs(l[0]-l[-1])
        path_len = sum([abs(l[i]-l[i-1]) for i in range(1, n_epochs)])
        return change/l[0], path_len/l[0]

    def epochs_since_best(self):
        ''' Number of epochs since the last best validate loss. '''
        return len(self.validate_loss_log)-1-self.best_index

    def print_best(self, text_io):
        print("Best epoch is %d with loss %.3f." % (self.best_index+1, self.best_loss), file=text_io)

####################################################################################################
