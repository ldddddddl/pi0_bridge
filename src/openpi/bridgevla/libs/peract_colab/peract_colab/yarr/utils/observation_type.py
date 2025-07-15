# From https://github.com/stepjam/YARR/blob/main/yarr/utils/observation_type.py


import numpy as np


class ObservationElement:
    def __init__(self, name: str, shape: tuple, type: type[np.dtype]):
        self.name = name
        self.shape = shape
        self.type = type
