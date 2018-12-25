import numpy as np
from gym.spaces.discrete import Discrete
from gym.spaces.box import Box
from gym.spaces.tuple_space import Tuple
from utils.numpy_extension import flatten, ndargmax


class SpaceGrid:
    """
    allows more flexible access to numpy array
    openai gym friendly
    Don't deal with multidimensionnal boxes.
    This is for storing whatever you want with keys action, value
    """

    def __init__(self,
                 resolution=None,
                 env=None,
                 gym_space=None,
                 boxes_resolution=None,
                 gym_spaces=None):
        # First possibility : creation by dimension
        self.dim_info = []  # info for each dimension (box or discrete, ...)
        if resolution is not None:
            self.shape = tuple(resolution)
            self.dim_info = [dict(type='discrete', resolution=shape_i) for shape_i in self.shape]

        # Second possibility: creation by env
        if env is not None:
            gym_spaces=(env.observation_space, env.action_space)
        elif gym_space is not None:
            gym_spaces = (gym_space,)
        assert gym_spaces is not None, 'Please give values for initialization'
        spaces_list = []
        for s in gym_spaces:
            spaces_list.extend(s.spaces) if isinstance(s, Tuple) else spaces_list.append(s)
        boxes_dimensions = [s.shape[0] for s in spaces_list if isinstance(s, Box)]
        if isinstance(boxes_resolution, tuple) and sum(boxes_dimensions) != len(boxes_resolution):
            raise ValueError('Resolution has wrong number of dimensions.')
        # now we can pass an int as resolution.
        if isinstance(boxes_resolution, int):
            boxes_resolution = sum(boxes_dimensions) * (boxes_resolution,)
        ri = 0  # Resolution index
        for s in gym_spaces:
            if isinstance(s, Discrete):
                self.dim_info.append(dict(type='discrete', resolution=s.n))
            elif isinstance(s, Box):
                nb_dims = s.shape[0]
                self.dim_info.extend([dict(type='continuous', resolution=boxes_resolution[ri + i],
                                           low=s.low[i], high=s.high[i]) for i in range(nb_dims)])
                ri += nb_dims
        self.shape = tuple([d['resolution'] for d in self.dim_info])
        self.array = np.random.rand(*self.shape)*1e-6

    def _coords(self, item):
        item = flatten(item) # standard utils library
        assert len(item) <= len(self.shape), 'Too many values in access key'
        assert None not in item, 'Corrupted Access key'
        coords = []
        for (i, c) in enumerate(item):
            dim = self.dim_info[i]
            if dim['type'] == 'discrete':
                assert c == int(c), 'Can\'t access to discrete dimension with non-int key'
                good_c = c
            elif dim['type'] == 'continuous':
                good_c = dim['resolution'] * (c - dim['low']) / (dim['high'] - dim['low'])
            assert 0 <= good_c <= dim['resolution']+1e-3, 'Key out of bounds :{0} for value {1}'.format(item, c)
            if dim['resolution'] <= good_c <= dim['resolution'] + 1:
                good_c = dim['resolution'] - 1
            coords.append(int(good_c))
        return tuple(coords)

    def __getitem__(self, item):
        return self.array[self._coords(item)]

    def __setitem__(self, key, value):
        self.array[self._coords(key)] = value

    def argmax(self, key):
        coords = ndargmax(self[key])
        res = []
        offset = len(key)
        for i,c in enumerate(coords):
            dim = self.dim_info[i+offset]
            if dim['type'] == 'discrete':
                res.append(c)
            elif dim['type'] == 'continuous':
                res.append(dim['low'] + (dim['high']-dim['low'])*c/(dim['resolution']-1))
        if len(res) == 1:
            return res[0]
        return res


