import numpy as np

class Logger:
    def __init__(self, callback=None):
        # Send To is for variable monitoring
        self.callback = callback
        self.buffer = {}

    def log(self, var, value):
        if not var in self.buffer.keys():
            self.buffer[var] = []
        self.buffer[var].append(value)
        self.callback.log(var, value) # to say that we are updating var with value


    def pop_buffer(self, var):
        if not var in self.buffer.keys():
            self.log('error', 'Attempted to access unknown key ({0} in Log'.format(var))
            var_buffer = []
        else:
            var_buffer = self.buffer[var]
            self.buffer[var] = []
        return var_buffer

    # standard logging
    def log_img(self, img):
        self.log('img', img)

    def log_error(self, err_string):
        self.log('error', err_string)