import numpy as np
import math
from policies.policy import Policy

# --- LA Policy (Lookahead) with Piecewise Costs ---
class LAPolicy_C(Policy):
    def __init__(self, params):
        """
        Lookahead (LA) policy for multi-location transshipment problems.
        Greedily transships one unit at a time if the expected future
        profit increase exceeds the marginal cost. This version supports
        piecewise concave transshipment costs.

        Args:
            params (dict): Simulation parameters including L, T, h, p, rho,
                           and piecewise cost parameters (tc_u, tc_m).
        """
        self.params = params
        self.L = params['L']
        self.T = params['T']
        self.rho = np.array(params['rho'])
        self.p = np.array(params['p'])
        self.h = np.array(params['h'])
        self.c = params['c'] # Base cost multiplier

        # expected_demands (np.ndarray): L x T matrix
        self.expected_demand_matrix = params['expected_demand_matrix']

        # --- START: Piecewise Cost Integration ---
        raw_tc_u = params.get('tc_u', params.get('trans_cost_breakpoints'))
        raw_tc_m = params.get('tc_m', params.get('trans_cost_marginals'))

        # Default to simple linear cost if not provided
        if raw_tc_u is None or raw_tc_m is None:
            raw_tc_u = [0.0]
            raw_tc_m = [1.0]

        # Normalize breakpoints to be cumulative, starting at 0
        tc_u = list(raw_tc_u)
        if not tc_u or tc_u[0] != 0.0:
            tc_u.insert(0, 0.0)

        # Ensure tc_u is strictly non-decreasing
        for i in range(1, len(tc_u)):
            if tc_u[i] < tc_u[i-1]:
                raise ValueError("tc_u must be non-decreasing cumulative breakpoints.")

        tc_m = list(raw_tc_m)
        expected_K = max(1, len(tc_u) - 1)
        if len(tc_m) < expected_K:
            last_marginal = tc_m[-1] if tc_m else 1.0
            tc_m.extend([last_marginal] * (expected_K - len(tc_m)))
        elif len(tc_m) > expected_K:
            tc_m = tc_m[:expected_K]

        self.tc_u = tc_u
        self.tc_m = tc_m
        self.K = len(self.tc_m)

        # Validate concavity: non-increasing marginals
        for a, b in zip(self.tc_m, self.tc_m[1:]):
            if a < b - 1e-12:
                raise ValueError("tc_m must be non-increasing (decreasing marginal cost).")
        # --- END: Piecewise Cost Integration ---

    def _get_marginal_cost(self, i, j, current_qty):
        """
        Compute the marginal cost to ship the *next* (qty+1)th unit from i->j.
        """
        if self.rho[i][j] == 0:
            return 0.0

        # Find which segment the next unit falls into
        marginal = self.tc_m[-1] # Default to the last marginal cost
        for k in range(self.K):
            seg_end = self.tc_u[k+1] if (k+1) < len(self.tc_u) else float('inf')
            if current_qty < seg_end:
                marginal = self.tc_m[k]
                break
        
        return self.c * self.rho[i][j] * marginal

    def _expected_future_profit_single_loc(self, y_i, i, current_t, total_T):
        """
        Recursively calculates the sum of expected single-period profits for
        a single location (i) starting with inventory y_i at current_t
        until total_T-1, assuming no transshipments and using expected demands.
        """
        y_i = int(round(y_i))
        if current_t >= total_T:
            return 0.0

        expected_d_t = self.expected_demand_matrix[i, current_t]
        expected_sales_t = min(y_i, expected_d_t)
        expected_holding_t = max(y_i - expected_d_t, 0.0)
        immediate_profit = self.p[i] * expected_sales_t - self.h[i] * expected_holding_t
        next_y_i = max(y_i - expected_d_t, 0.0)
        return immediate_profit + self._expected_future_profit_single_loc(next_y_i, i, current_t + 1, total_T)

    def __call__(self, x, t, d):
        """
        Determines the transshipment decision for the current state (x) and time (t)
        using the greedy one-unit lookahead policy with marginal costs.
        """
        L = self.L
        z = np.zeros((L, L), dtype=int)
        y_current = np.array(x, dtype=float)

        while True:
            best_net_benefit = 1e-9
            best_move = None

            for i in range(L):
                for j in range(L):
                    if i == j or y_current[i] <= 0:
                        continue

                    y_after_move = y_current.copy()
                    y_after_move[i] -= 1
                    y_after_move[j] += 1

                    value_current = sum(self._expected_future_profit_single_loc(y_current[k], k, t, self.T) for k in range(L))
                    value_after_move = sum(self._expected_future_profit_single_loc(y_after_move[k], k, t, self.T) for k in range(L))

                    benefit = value_after_move - value_current
                    
                    # --- COST CALCULATION CHANGE ---
                    # Calculate cost based on the *next* unit to be shipped on this lane (i, j)
                    cost = self._get_marginal_cost(i, j, z[i, j])
                    # --- END CHANGE ---

                    net_benefit = benefit - cost

                    if net_benefit > best_net_benefit:
                        best_net_benefit = net_benefit
                        best_move = (i, j)

            if best_move is not None:
                i, j = best_move
                z[i][j] += 1
                y_current[i] -= 1
                y_current[j] += 1
            else:
                break

        return z.astype(int).tolist()
