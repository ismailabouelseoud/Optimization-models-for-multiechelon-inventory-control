import numpy as np
import itertools
from policies.policy import Policy
# ----------------------------
# 1. DP POLICY (BACKWARD DP)
# ----------------------------
class DPGenPolicy(Policy):
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
        self._compute_dp_gen()

    def model(self):
        """
        Returns the model of the DP policy.
        """
        
        '''
        max_inv = sum(self.S)
        x = [x1, x2, x3]  # Inventory levels at each location
        z[i][j] = [ [z11, z12, z13], [z21, z22, z23], [z31, z32, z33] ]  # Orders placed at each location
        Need to calculate all the possible values of z[i][j] 

        y[i] = x[i] + sum([z[i][j] - z[j][i] for j in range(self.L)]) 
        x_post[i] = max(0, y[i] - d[i])  # Update inventory after demand

        cost = sum([c * rho[i][j] * z[i][j] for i in range(self.L) for j in range(self.L)])
        reward = sum([p[i] * min(y[i], d[i]) for i in range(self.L)]) - sum([h[i] * max(y[i] - d[i], 0) for i in range(self.L)])
        profit = reward - cost
        From a set of policies fin the one that maximizes sum of profits over all periods...
        is a stochastic optimization problem, which has a non-linear structure and accepts integer variables only.
        Turn it into a recurive linear programming problem:
        V_t(x, t) = max_{z}[ -const + V_z{y, t+1}(x_post, t+1)]
        where:
        V_z(y, t) =  max_{z} E[reward + V{t+1}(x_post, t+1)]
        
        '''
        pass

    def _compute_dp_gen(self):
        """
        Compute DP tables for L-location problem by enumerating all feasible
        transshipment splits.
        """
        # Helper: generate all k-tuples of non-negative ints summing to n
        def all_splits(n, k):
            # Uses stars-and-bars via combinations of divider positions
            for dividers in itertools.combinations(range(n + k - 1), k - 1):
                result = []
                last = 0
                for divider in dividers:
                    result.append(divider - last)
                    last = divider + 1
                result.append(n + k - 1 - last)
                yield tuple(result)

        max_inv = sum(self.S)  # Maximum total inventory across all locations

        # Backward induction over periods
        for t in reversed(range(self.T)):
            # All possible inventory levels per location
            x_ranges = [range(self.S[i] + 1) for i in range(self.L)]
            # Iterate every joint state x = (x1, x2, ..., xL)
            for x in itertools.product(*x_ranges):
                best_val = -np.inf
                best_z = None

                # Precompute possible z-rows for each origin location
                # z_rows[i] lists all ways to split x[i] units to L destinations
                all_z_rows = [list(all_splits(x[i], self.L)) for i in range(self.L)]

                # Enumerate every transshipment matrix z via Cartesian product
                for z_tuple in itertools.product(*all_z_rows):
                    # Convert tuple rows to matrix form
                    z = [list(row) for row in z_tuple]

                    # Compute post-transshipment inventory y for each location
                    y = []
                    for i in range(self.L):
                        inflow = sum(z[j][i] for j in range(self.L))
                        outflow = sum(z[i][j] for j in range(self.L))
                        y.append(x[i] + inflow - outflow)

                    # Total transshipment cost across all origin-destination pairs
                    cost = sum(
                        self.c * self.rho[i][j] * z[i][j]
                        for i in range(self.L)
                        for j in range(self.L)
                    )

                    # Expected value accumulator
                    ev = 0.0
                    # Loop over demand scenarios in PMF
                    for d, prob in self.demand_pmf.items():
                        reward = 0
                        x_post = [0] * self.L
                        # Calculate reward and next inventory for each location
                        for i in range(self.L):
                            # Sales revenue minus holding cost
                            reward += (
                                self.p[i] * min(y[i], d[i])
                                - self.h[i] * max(y[i] - d[i], 0)
                            )
                            # Inventory carried to next period (capped)
                            x_post[i] = min(max(y[i] - d[i], 0), max_inv)

                        ev += prob * (reward + self.V[t + 1][x_post[0]][x_post[1]])

                    # Net value after subtracting shipping cost
                    val = ev - cost

                    # Update best decision if improved
                    if val > best_val:
                        best_val = val
                        best_z = z

                # Store computed value and optimal action
                self.V[t][x[0]][x[1]] = best_val
                self.best_action[t][x[0]][x[1]] = best_z

    def __call__(self, x, t, d):
        """
        Retrieve the precomputed optimal transshipment plan for state x at period t.
        Returns an LxL matrix z of shipment quantities.
        """
        z = self.best_action[t][x[0]][x[1]]
        return z