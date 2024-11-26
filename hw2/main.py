import matplotlib.pyplot as plt
import numpy as np

class ForestField:
    def __init__(self, n, p = 0.3) -> None:
        self.n = n
        self.field = np.zeros((self.n, self.n), dtype=np.int32)
        self.p = p
        
    def step(self):
        rand_vals = np.random.rand(*self.field.shape)
        
        self.field[rand_vals > self.p and self.field == 1] = -1
        self.fireSpread()
        self.field[rand_vals < self.p and self.field == 0]
        
    def fireSpread(self):
        pass
        
        
    