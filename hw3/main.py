import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt

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
    fig, ax = plt.subplots()
    while itmo_clubs.peopleNum < 100:
        itmo_clubs.step()
    ax.bar((range(len(itmo_clubs.clubs))), itmo_clubs.clubs, color='blue')
    ax.set_xlabel('Номер клуба')
    ax.set_ylabel('Число людей в клубе')
    fig.suptitle(f"Число студентов: {itmo_clubs.peopleNum}")
    # plt.ylim(0, np.max(itmo_clubs.clubs) * 1.05)
    plt.show()
    # plt.pause()
    # plt.cla()
    return

if __name__ == '__main__':
    main()