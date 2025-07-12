import numpy as np
from policies.policy import Policy

# --- LA Policy (Lookahead) ---
class LAPolicy(Policy):
    def __init__(self, params):
        """
        Lookahead (LA) policy for multi-location transshipment problems.
        Greedily transships one unit at a time if the expected future
        profit increase exceeds the cost, assuming no future transshipments
        after the current period's decision.

        Args:
            params (dict): Simulation parameters including L, T, h, p, c, rho.
            
        """
        self.params = params
        self.L = params['L']
        self.T = params['T']
        self.rho = np.array(params['rho']) # Ensure rho is numpy array
        self.p = np.array(params['p'])     # Ensure p is numpy array
        self.h = np.array(params['h'])     # Ensure h is numpy array
        self.c = params['c']

        # expected_demands (np.ndarray): L x T matrix where element (i, t)
        # is the expected demand for location i in period t.
        self.expected_demand_matrix = params['expected_demand_matrix']


    def _expected_future_profit_single_loc(self, y_i, i, current_t, total_T):
        """
        Recursively calculates the sum of expected single-period profits for
        a single location (i) starting with inventory y_i at current_t
        until total_T-1, assuming no transshipments and using expected demands.
        """
        # Ensure y_i is treated as an integer for inventory levels
        y_i = int(round(y_i)) # Round to nearest integer inventory level

        if current_t >= total_T:
            return 0.0

        # Get expected demand for the current period
        expected_d_t = self.expected_demand_matrix[i, current_t]

        # Calculate expected immediate profit for this period
        expected_sales_t = min(y_i, expected_d_t)
        expected_holding_t = max(y_i - expected_d_t, 0.0) # Use 0.0 for float consistency
        immediate_profit = self.p[i] * expected_sales_t - self.h[i] * expected_holding_t

        # Calculate next period's expected starting inventory (after demand)
        next_y_i = max(y_i - expected_d_t, 0.0) # Use 0.0

        # Recursive step: Add immediate profit and expected future profit from the next state
        return immediate_profit + self._expected_future_profit_single_loc(next_y_i, i, current_t + 1, total_T)


    def __call__(self, x, t, d):
        """
        Determines the transshipment decision for the current state (x) and time (t)
        using the greedy one-unit lookahead policy.

        Args:
            x (list or np.ndarray): Current inventory levels at each location.
            t (int): Current time period (0 to T-1).

        Returns:
            list of lists: The transshipment matrix z[i][j].
        """
        L = self.L
        # Initialize transshipment matrix to zeros
        z = np.zeros((L, L), dtype=int)

        # Use a working copy of the current inventory (will be updated by transshipments)
        y_current = np.array(x, dtype=float) # Use float for intermediate calculations

        # Greedily add transshipments one unit at a time as long as it's beneficial
        while True:
            best_net_benefit = 1e-9 # Initialize with a tiny positive value to allow first beneficial move
            best_move = None # Stores (source_idx, dest_idx)

            # Iterate through all possible pairs of source (i) and destination (j) locations
            for i in range(L):
                for j in range(L):
                    # Cannot transship to the same location or from a location with no stock left to ship
                    # Stock available to ship from i is its current temp level y_current[i]
                    if i == j or y_current[i] <= 0:
                        continue

                    # Calculate the state after hypothetical transshipment of 1 unit from i to j
                    # This only affects the inventory levels at i and j for the purpose of evaluation
                    # The actual z matrix is updated only after a move is selected
                    y_after_move = y_current.copy()
                    y_after_move[i] -= 1
                    y_after_move[j] += 1

                    # Calculate the total expected future profit from the current state (y_current)
                    # This is the sum of expected future profits for each location independently
                    value_current = sum(self._expected_future_profit_single_loc(y_current[k], k, t, self.T) for k in range(L))

                    # Calculate the total expected future profit from the state after the move (y_after_move)
                    value_after_move = sum(self._expected_future_profit_single_loc(y_after_move[k], k, t, self.T) for k in range(L))

                    # Calculate the benefit of this single unit move (increase in expected future profit)
                    benefit = value_after_move - value_current

                    # Calculate the direct transshipment cost
                    cost = self.c * self.rho[i][j]

                    # Calculate the net benefit
                    net_benefit = benefit - cost

                    # Check if this move is the best beneficial move found so far
                    if net_benefit > best_net_benefit:
                        best_net_benefit = net_benefit
                        best_move = (i, j)

            # If a beneficial move was found (net_benefit > 0)
            if best_move is not None:
                # Perform the best move by updating the transshipment matrix and current inventory levels
                i, j = best_move
                z[i][j] += 1
                y_current[i] -= 1
                y_current[j] += 1
            else:
                # No beneficial move found, stop adding transshipments
                break

        # Return the final transshipment matrix as a list of lists of integers
        return z.astype(int).tolist()


