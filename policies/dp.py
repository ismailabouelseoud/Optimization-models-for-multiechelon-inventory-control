import numpy as np
from policies.policy import Policy
# ----------------------------
# 1. DP POLICY (BACKWARD DP)
# ----------------------------
class DPPolicy(Policy):
    """
    Dynamic Programming Policy for a two-product inventory system.
    """
    def __init__(self, params):
        """
        Initializes the DP policy with the given parameters.
        Args:
            params (dict): Dictionary containing the parameters for the policy.
                Expected keys:
                - 'L': Number of locations. (Practically unused in this implementation)
                - 'S': List of initial inventory levels for each location ( of length 2)
                - 'T': Total number of time periods. 
                - 'h': Holding costs for each product.
                - 'p': Prices for each product.
                - 'c': Cost of ordering each product.
                - 'rho': Distance matrix.
                - 'full_demand_matrix': Demand probability mass function as a dictionary.
        """
        self.L = params['L']
        self.S = params['S']
        self.T = params['T']
        self.h = params['h']
        self.p = params['p']
        self.c = params['c']
        self.rho = params['rho']
        self.demand_pmf = params['full_demand_matrix']
        max_inv = self.S[0] + self.S[1]
        # Expanded DP tables to avoid out-of-bounds
        self.V = [ np.zeros((max_inv+1, max_inv+1)) for _ in range(self.T+1) ]
        self.best_action = [
            [[None]*(max_inv+1) for _ in range(max_inv+1)]
            for _ in range(self.T)
        ]
        self._compute_dp()

    def _compute_dp(self):
        max_inv = self.S[0] + self.S[1]
        for t in reversed(range(self.T)):
            for x1 in range(max_inv+1):
                for x2 in range(max_inv+1):
                    best_val = -1e9
                    best_z = None
                    # Enumerate all feasible z_ijt
                    for z11 in range(x1+1):
                        z12 = x1 - z11
                        for z21 in range(x2+1):
                            z22 = x2 - z21
                            y1 = z11 + z21
                            y2 = z12 + z22
                            cost = self.c * (self.rho[0][1] * z12 + self.rho[1][0] * z21)
                            ev = 0.0
                            for (d1,d2), prob in self.demand_pmf.items():
                                sold1 = min(y1,d1)
                                sold2 = min(y2,d2)
                                reward = (self.p[0] * sold1 + self.p[1] * sold2
                                          - self.h[0]*max(y1 - d1, 0)
                                          - self.h[1]*max(y2 - d2, 0))
                                x1p = min(max(y1 - d1, 0), max_inv)
                                x2p = min(max(y2 - d2, 0), max_inv)
                                ev += prob * (reward + self.V[t + 1][x1p][x2p])
                            val = ev - cost
                            if val > best_val:
                                best_val = val
                                best_z = (z11, z12, z21, z22)
                    self.V[t][x1][x2] = best_val
                    self.best_action[t][x1][x2] = best_z


    def __call__(self, x, t, d):

        z11, z12, z21, z22 = self.best_action[t][x[0]][x[1]]
        return [[z11, z12], [z21, z22]]
