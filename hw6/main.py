import numpy as np
import matplotlib.pyplot as plt
class BananaRepublic:
    def __init__(self, s = 0.2, d = 0.1, workforce = 100, M_0 = 100):
        self.s = s
        self.d = d
        self.workforce = workforce
        self.M = []
        if M_0 is not None:
            self.M.append(M_0)
        self.O = []
        self.I = []
        self.C = []
        self.time = [0]
         
    def update(self):
        self.O.append(int(self.workforce * np.sqrt(self.M[-1])))
        self.I.append(self.s * self.O[-1])
        self.C.append(self.O[-1] - self.I[-1])
        self.M.append(int(self.M[-1] * (1 - self.d) + self.I[-1]))
        if len(self.time) == 0:
            self.time.append(0)
        else:
            self.time.append(self.time[-1] + 1)
            
rep = BananaRepublic()

for _ in range(200):
    rep.update()
    
plt.plot(rep.time, rep.M, label="Кол-во машин")
plt.plot(rep.time[:-1], rep.O, label="Объём производства")
plt.plot(rep.time[:-1], rep.I, label="Инвестиции")
plt.legend()
plt.show()
            