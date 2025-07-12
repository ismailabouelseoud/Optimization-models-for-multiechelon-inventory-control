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
        """
        Compute the dynamic programming (DP) tables for a two-location transshipment problem.
        """
        # Maximum total inventory possible across both locations
        max_inv = self.S[0] + self.S[1]

        # Backward induction from final period to period 0
        for t in reversed(range(self.T)):
            # Loop over all inventory levels at loc1 (x1) and loc2 (x2)
            for x1 in range(max_inv + 1):
                for x2 in range(max_inv + 1):
                    best_val = -np.inf  # Initialize best value
                    best_z = None       # Placeholder for optimal decision tuple

                    # Enumerate all feasible transshipment splits z_ij:
                    # z11,z12 from loc1's x1; z21,z22 from loc2's x2
                    for z11 in range(x1 + 1):
                        z12 = x1 - z11
                        for z21 in range(x2 + 1):
                            z22 = x2 - z21
                            # y1: inventory at loc1 after transshipment in period t
                            y1 = z11 + z21
                            # y2: inventory at loc2 after transshipment
                            y2 = z12 + z22

                            # Transshipment cost: c * distance * quantity shipped
                            cost = (
                                self.c * (
                                    self.rho[0][1] * z12 +  # ship from 1->2
                                    self.rho[1][0] * z21    # ship from 2->1
                                )
                            )

                            # Expected value accumulator
                            ev = 0.0

                            # Sum over all possible demand realizations
                            for (d1, d2), prob in self.demand_pmf.items():
                                # Units sold = min(inventory, demand)
                                sold1 = min(y1, d1)
                                sold2 = min(y2, d2)

                                # Immediate reward: revenue minus holding cost
                                reward = (
                                    self.p[0] * sold1 + self.p[1] * sold2  # revenue from sales
                                    - self.h[0] * max(y1 - d1, 0)           # holding cost at loc1
                                    - self.h[1] * max(y2 - d2, 0)           # holding cost at loc2
                                )

                                # Next period inventory: leftover stock, capped at max_inv
                                x1p = min(max(y1 - d1, 0), max_inv)
                                x2p = min(max(y2 - d2, 0), max_inv)

                                # Add discounted future value from V at t+1
                                ev += prob * (reward + self.V[t + 1][x1p][x2p])

                            # Net value after subtracting transshipment cost
                            val = ev - cost

                            # Update best decision if value improves
                            if val > best_val:
                                best_val = val
                                best_z = (z11, z12, z21, z22)

                    # Store optimal value and action in DP tables
                    self.V[t][x1][x2] = best_val
                    self.best_action[t][x1][x2] = best_z

    def __call__(self, x, t, d):
        """
        Return the precomputed optimal transshipment action for state (x, t).
        x: current inventory vector [x1, x2]
        t: current period index
        d: observed demand (unused here, as action is determined before demand)
        """
        # Retrieve optimal z-split and format as 2x2 matrix
        z11, z12, z21, z22 = self.best_action[t][x[0]][x[1]]
        return [[z11, z12], [z21, z22]]