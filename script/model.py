import numpy as np
import pickle
import six
import chainer
import chainer.functions as F

class Model(chainer.FunctionSet):
    def __init__(self, **functions):
        chainer.FunctionSet.__init__(self, **functions)

    @classmethod
    def load(self, file_path):
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model

    def save(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f, -1)

    def forward(self, x, train=True):
        raise "forward must be implemented"
