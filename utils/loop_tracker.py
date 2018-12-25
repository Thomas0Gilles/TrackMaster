import time

class LoopTracker:
    '''
    Loop tracker is for monitoring a loop that takes a loong time.
    you should track the loop variable every iteration
    '''
    def __init__(self, nb_iterations, frq=5):
        self.frq = int(frq)
        print('Initializing Tracker to tick every {0}%'.format(self.frq))
        print('# Loop Iterations : ', nb_iterations)
        self.nb_iterations = nb_iterations
        self.start_time = time.time()
        self.completion = 0

    def track(self, i):
        p = 100*(i+1)/self.nb_iterations
        if p>self.completion+self.frq:
            self.completion += self.frq
            dt = time.time() - self.start_time
            print('Loop Completion : {0}%, Remaining Time : {1}s'.format(
                self.completion, int(dt/(i+1)*(self.nb_iterations-i))))
