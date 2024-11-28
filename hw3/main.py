import numpy as np
import random
import time

class ClubSystem:
    
    def __init__(self, p = 0.1, clubs = []):
        self.p = p
        self.clubs = clubs
        self.peopleNum = 0
        self.weights = []
        if len(self.clubs) > 0:
            self.peopleNum = sum(self.clubs)
        self.weights = self.updateWeights()
        
    def step(self):
        if np.random.rand() < self.p or len(self.clubs) == 0:
            self.clubs.append(1)
        else:
            club = random.choices([k for k in range(len(self.clubs))], weights=self.weights, k = 1)[0]
            self.clubs[club] += 1
        self.peopleNum += 1
        self.updateWeights()
            
    def updateWeights(self):
        if len(self.clubs) == 0:
            self.weights = []
            return
        self.weights = [k / self.peopleNum for k in self.clubs]



def main():
    itmo_clubs = ClubSystem(p = 0.4)
    for _ in range(20):
        itmo_clubs.step()
        print(itmo_clubs.clubs)
        time.sleep(1)
    return

if __name__ == '__main__':
    main()