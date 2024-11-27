import matplotlib.pyplot as plt
import numpy as np
import os
import time
def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

class ForestField:
    def __init__(self, n=5, p = 0.3) -> None:
        self.n = n
        self.field = np.zeros((self.n, self.n), dtype=np.int32)
        self.p = p
        
    def step(self):
        rand_vals = np.random.rand(*self.field.shape)
        
        #lightning strike
        self.field[np.logical_and(rand_vals > self.p, self.field == 1)] = -1
        clear_console()
        print(self.field)
        time.sleep(1)
        #fire spread
        self.fireSpread()
        clear_console()
        print(self.field)
        time.sleep(1)
        #fire extinguish
        self.field[self.field == -1] = 0
        clear_console()
        print(self.field)
        time.sleep(1)
        #forest growth
        self.field[rand_vals < self.p] = 1
        clear_console()
        print(self.field)
        time.sleep(1)
        
    def fireSpread(self):
        dx = [-1, 0, 1, 0]
        dy = [0, 1, 0, -1]
        for i in range(self.n):
            for j in range(self.n):
                neighbours = []
                for k in range(len(dx)):
                    if not ( 0 <= i + dx[k] < self.n and 0 <= j + dy[k] < self.n):
                        continue
                    if self.field[i + dx[k]][j + dy[k]] == 0 or self.field[i][j] == 0:
                        continue
                    neighbours.append(self.field[i + dx[k]][j + dy[k]])
                if len(neighbours) > 0:
                    self.field[i][j] = min(neighbours)
                    
n = 5
forest = ForestField(n)

for i in range(10):
    forest.step()