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
         
    def step(self, burnAll = False):
        dirt = 0
        fire = -1
        tree = 1
        cell = (np.random.randint(0, self.n), np.random.randint(0, self.n))
        rand_val = np.random.rand()
        self.draw(img)
        #forest growth
        if self.field[cell[0], cell[1]] == dirt and rand_val < self.p:
                self.field[cell[0], cell[1]] = 1
        self.treesPreFire.append(self.countTrees())
        #lightning strike
        if self.field[cell[0], cell[1]] == tree and rand_val >= self.p:
            self.field[cell[0], cell[1]] = -1
        self.draw(img)
        #fire spread
        self.fireSpread(burnAll)
        self.draw(img)
        #fire extinguish
        self.field[self.field == -1] = 0
        self.draw(img)
        self.treesPostFire.append(self.countTrees())
        self.draw(img)
        
    def fireSpread(self, burnAll = False):
        dx = [-1, 0, 1, 0, -1, -1, 1, 1]
        dy = [0, 1, 0, -1, -1, 1, 1, -1]
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
            if not burnAll:
                return
        
    def countTrees(self):
        return np.sum(self.field[self.field == 1])
            
    def draw(self, img):
        img.set_array(self.field)
        plt.draw()
        plt.pause(0.1)
    
n = 30
forest = ForestField(n, p=9)


plt.ion()  # Turn on interactive mode
fig, ax = plt.subplots()
cmap = ListedColormap(['red', '#9f3b00', 'green'])
img = ax.imshow(forest.field, cmap=cmap, interpolation='nearest')
plt.colorbar(img)

for i in range(1000):
    forest.step(burnAll=True)
    
plt.ioff()