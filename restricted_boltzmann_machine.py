import numpy as np

class RestrictedBoltzmannMachine:

    def __init__(self, n_visible, n_hidden):
        self.weights = np.random.randn(n_visible, n_hidden)
        self.visible = np.zeros(n_visible)
        self.hidden = np.zeros(n_hidden)

    def train(self, data, epochs=100):
        for k in range(epochs):
            