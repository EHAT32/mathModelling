import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.colors import ListedColormap
def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')

class ForestField:
    def __init__(self, n=5, p = 0.6) -> None:
        self.n = n
        self.field = np.zeros((self.n, self.n), dtype=np.int32)
        self.p = p
        self.treesPreFire = []
        self.treesPostFire = []
        self.dirt = 0
        self.tree = 1
        self.fire = -1
         
    def step(self, burnAll = False):
        cell = (np.random.randint(0, self.n), np.random.randint(0, self.n))
        rand_val = np.random.rand()
        self.draw(img)
        #forest growth
        if self.field[cell[0], cell[1]] == self.dirt and rand_val < self.p:
                self.field[cell[0], cell[1]] = 1
        self.treesPreFire.append(self.countTrees())
        #lightning strike
        if self.field[cell[0], cell[1]] == self.tree and rand_val >= self.p:
            self.field[cell[0], cell[1]] = -1
        self.draw(img)
        #fire spread
        self.fireSpread(burnAll)
        self.draw(img)
        #fire extinguish
        self.field[self.field == self.fire] = self.dirt
        self.draw(img)
        self.treesPostFire.append(self.countTrees())
        self.draw(img)
        
    def fireSpread(self, burnAll = False):
        dx = [-1, 0, 1, 0, -1, -1, 1, 1]
        dy = [0, 1, 0, -1, -1, 1, 1, -1]
        old_field = None
        while old_field is None or (old_field != self.field).any():
            old_field = self.field.copy()
            fire_indices = np.where(self.field == self.fire)
            fire_indices = list(zip(fire_indices[0], fire_indices[1]))
            for idx in fire_indices:
                i, j = idx
                for k in range(len(dx)):
                    if not ( 0 <= i + dx[k] < self.n and 0 <= j + dy[k] < self.n):
                        continue
                    if self.field[i + dx[k]][j + dy[k]] == self.dirt:
                        continue
                    self.field[i + dx[k]][j + dy[k]] = self.fire
            if not burnAll:
                return
        
    def countTrees(self):
        return np.sum(self.field[self.field == self.tree])
            
    def draw(self, img):
        img.set_array(self.field)
        plt.draw()
        plt.pause(0.1)
    
n = 10
forest = ForestField(n, p=0.9)


plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
cmap = ListedColormap(['red', '#9f3b00', 'green'])
img = ax.imshow(forest.field, cmap=cmap, interpolation='nearest')
plt.colorbar(img)

for i in range(1000):
    forest.step(burnAll=True)
    
plt.ioff()