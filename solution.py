import numpy as np


class Solution:
    def __init__(self, T0, Tleft, Tright, Ntime, Nspace, dx) -> None:
        self.solution = np.zeros((Ntime, Nspace))
        self.Ntime = Ntime
        self.Nspace = Nspace
        self.Tleft = Tleft
        self.Tright = Tright
        self.solution[0, :] = T0
        self.timestep = 0
        self.dx = dx
        self.applyRB()

    def newTime(self):
        self.timestep = self.timestep + 1
        try:
            self.solution[self.timestep, :] = self.solution[self.timestep - 1, :]
        except IndexError:
            print("maximum timestep reached")
        self.applyRB()

    def applyRB(self):
        self.solution[self.timestep, 0] = self.Tleft
        self.solution[self.timestep, -1] = self.Tright

    def getVal(self, spaceiter):
        if spaceiter < 0:
            spaceiter = 0
        elif spaceiter > self.Nspace - 1:
            spaceiter = self.Nspace - 1
        return self.solution[self.timestep, spaceiter]

    def getPreVal(self, spaceiter):
        if spaceiter < 0:
            spaceiter = 0
        elif spaceiter > self.Nspace - 1:
            spaceiter = self.Nspace - 1
        return self.solution[self.timestep - 1, spaceiter]

    def setVal(self, spaceiter, value):
        if spaceiter <= 0:
            pass
        elif spaceiter >= self.Nspace - 1:
            pass
        else:
            self.solution[self.timestep, spaceiter] = value[0, 0]

    def getTimestep(self):
        return self.solution[self.timestep, :]

    def getPreTimestep(self):
        return self.solution[self.timestep - 1, :]
