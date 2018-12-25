from utils import ndargmax, flatten
import numpy as np
eps = 1e-6

class ContinuousStateActionMap:
    # for env that are Boxes only !
    def __init__(self, env, boxes_resolution=10):
        self.env = env
        self.space_nb_dims = env.observation_space.shape[0] + env.action_space.shape[0]
        self.low = np.concatenate((env.observation_space.low, env.action_space.low))
        self.high = np.concatenate((env.observation_space.high, env.action_space.high))
        if isinstance(boxes_resolution, int):
            boxes_resolution = self.space_nb_dims*[boxes_resolution]
        assert len(boxes_resolution) == self.space_nb_dims
        self.shape = np.array(boxes_resolution)
        self.array = np.random.rand(*self.shape)*eps

    def _coords(self, key):
        key = flatten(key)
        l = len(key)
        low, high = self.low[:l], self.high[:l]
        assert l <= len(self.shape), 'Too many values in access key'
        assert None not in key, 'Corrupted Access key'
        assert all(low-eps <= key) and all(key <= high+eps), 'Key out of bounds'+str(key)
        coords = (np.array(key)-eps-low)/(high - low)*self.shape[:l]
        return tuple(map(int, coords))

    def __getitem__(self, key):
        return self.array[self._coords(key)]

    def __setitem__(self, key, value):
        self.array[self._coords(key)] = value

    def argmax(self, key):
        coords = ndargmax(self[key])
        l = len(key)
        return self.low[l:] + (self.high[l:]-self.low[l:])*np.array(coords)/(self.shape[l:]-1)