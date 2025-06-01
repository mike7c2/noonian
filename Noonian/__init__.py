from queue import Queue

class QueueShutdownException(Exception):
    pass

class ShutdownQueue(Queue):
    """
    Derivative of queue implementing basic shutdown mechanism
    """
    def __init__(self):
        super(ShutdownQueue, self).__init__()

    def shutdown(self):
        super(ShutdownQueue, self).put(None)

    def get(self, block=True, timeout=None):
        x = super(ShutdownQueue, self).get(block, timeout)
        if x is None:
            raise QueueShutdownException()
        else:
            return x