import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import ListedColormap
def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

class ForestField:
    def __init__(self, n=5, p = 0.2) -> None:
        self.n = n
        self.field = np.zeros((self.n, self.n), dtype=np.int32)
        self.p = p
         
    def step(self):
        rand_vals = np.random.rand(*self.field.shape)
        self.draw(img)
        #lightning strike
        self.field[np.logical_and(rand_vals > self.p, self.field == 1)] = -1
        self.draw(img)
        #fire spread
        self.fireSpread()
        self.draw(img)
        #fire extinguish
        self.field[self.field == -1] = 0
        self.draw(img)
        #forest growth
        self.field[rand_vals < self.p] = 1
        self.draw(img)
        
    def fireSpread(self):
        dx = [-1, 0, 1, 0]
        dy = [0, 1, 0, -1]
        old_field = None
        while old_field is None or (old_field != self.field).any():
            old_field = self.field.copy()
            for i in range(self.n):
                for j in range(self.n):
                    for k in range(len(dx)):
                        if not ( 0 <= i + dx[k] < self.n and 0 <= j + dy[k] < self.n):
                            continue
                        if self.field[i + dx[k]][j + dy[k]] == 0 or self.field[i][j] == 0:
                            continue
                        if self.field[i + dx[k]][j + dy[k]] == -1:
                            self.field[i][j] = -1
        
    def countTrees(self):
        return np.sum(self.field[self.field == 1])
            
    def draw(self, img):
        img.set_array(self.field)
        plt.draw()
        plt.pause(0.5)
    
n = 60
forest = ForestField(n)


plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
cmap = ListedColormap(['red', '#9f3b00', 'green'])
img = ax.imshow(forest.field, cmap=cmap, interpolation='nearest')
plt.colorbar(img)

for i in range(1000):
    forest.step()