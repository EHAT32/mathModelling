import numpy as np
import matplotlib.pyplot as plt
class BananaRepublic:
    def __init__(self, s = 0.2, d = 0.1, workforce = 10000, M_0 = 100):
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
        self.O.append(int(np.sqrt(self.workforce * self.M[-1])))
        self.I.append(self.s * self.O[-1])
        self.C.append(self.O[-1] - self.I[-1])
        self.M.append(int(self.M[-1] * (1 - self.d) + self.I[-1]))
        if len(self.time) == 0:
            self.time.append(0)
        else:
            self.time.append(self.time[-1] + 1)
            
    def calcBalance(self):
        return self.workforce * self.s / self.d        
            
rep = BananaRepublic()

for _ in range(200):
    rep.update()
    
pred_line = [rep.calcBalance()] * len(rep.time)
    
plt.plot(rep.time[:-1], rep.O, label="Объём производства")
plt.plot(rep.time, pred_line, '--r', label=f'Баланс по Солоу: {int(rep.calcBalance())}')
plt.legend()
plt.show()
            