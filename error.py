class CleanError(BaseException):
    def __init__(self, message=None):
        if message is None:
            message = 'Please clean the spectrum by using the "clean" method'
        self.message = message


class NormalizeError(BaseException):
    def __init__(self, message=None):
        if message is None:
            message = 'Please normalize the spectrum by using the "normalize" method'
        self.message = message
