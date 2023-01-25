import time

####################################################################################################

class ExecutionTimer:
    ''' Timer class, useful for with statement. '''

    def __init__(self, console):
        self.start = None
        self.stop = None
        self.console = console

    def __enter__(self):
        self.start = time.time()
        if self.console:
            print("Starting at:", time.strftime("%H:%M:%S"))
        return self

    def __exit__(self, *exc):
        self.stop = time.time()
        if self.console:
            print(self.duration_str())
        return False

    def duration(self):
        return self.stop-self.start

    def duration_str(self):
        return "Duration: %.1fs"%self.duration()

####################################################################################################
