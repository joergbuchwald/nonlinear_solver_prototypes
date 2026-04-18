import numpy as np


def K(T):
    """
    heat conductivity
    """
    return np.abs(0.1 - T / 2000)


def F(solution, dt, dTleft=0.0, dTmiddle=0.0, dTright=0.0):
    """
    residual
    """
    R = []
    for spaceiter in range(0, solution.Nspace):
        Tnewleft = solution.getVal(spaceiter - 1) + dTleft
        Tnewright = solution.getVal(spaceiter + 1) + dTright
        Tnewmiddle = solution.getVal(spaceiter) + dTmiddle
        Toldmiddle = solution.getPreVal(spaceiter)
        R.append(
            K(solution.getVal(spaceiter))
            * dt
            * (Tnewleft - 2 * Tnewmiddle + Tnewright)
            / (solution.dx**2)
            - Tnewmiddle
            + Toldmiddle
        )
    R[-1] = 0
    R[0] = 0
    return np.matrix(R).transpose()


def JF(solution, dt):
    """
    Jacobian
    """
    dT = 1e-6
    J = np.matrix(np.zeros((solution.Nspace, solution.Nspace)))
    for i in range(solution.Nspace):
        for j in range(solution.Nspace):
            if j == i - 1:
                J[i, j] = (
                    F(solution, dt, dTleft=dT)[i, 0] - F(solution, dt, dTleft=0.0)[i, 0]
                ) / dT
            elif j == i:
                J[i, j] = (
                    F(solution, dt, dTmiddle=dT)[i, 0]
                    - F(solution, dt, dTmiddle=0.0)[i, 0]
                ) / dT
            elif j == i + 1:
                J[i, j] = (
                    F(solution, dt, dTright=dT)[i, 0]
                    - F(solution, dt, dTright=0.0)[i, 0]
                ) / dT
            else:
                J[i, j] = 0.0
    J[:, 0] = 0
    J[0, :] = 0
    J[:, -1] = 0
    J[-1, :] = 0
    J[0, 0] = 1
    J[-1, -1] = 1
    return J
