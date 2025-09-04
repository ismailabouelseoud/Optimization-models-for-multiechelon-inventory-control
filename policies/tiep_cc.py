import numpy as np
import math
from policies.policy import Policy

class TIEPolicy_C(Policy):
    def __init__(self, params, safety=0):
        """
        Initialize TIEPolicy with parameters and optional safety factor.
        Now includes support for piecewise concave transshipment costs.

        params keys:
            - 'L': number of locations
            - 'mu': vector of average demands per period
            - 'rho': distance matrix (L x L)
            - 'S': initial inventory levels (will be copied)
            - 'h': holding cost vector
            - 'p': penalty/backorder cost vector
            - 'T': time horizon
            - 'demand_sampler': function to sample demand
            - 'c': base multiplier for distance (float, optional, default=1.0)
            - 'tc_u' or 'trans_cost_breakpoints': list of cumulative breakpoints (start at 0)
            - 'tc_m' or 'trans_cost_marginals': marginals per segment
        """
        self.L = params['L']
        self.mu = params['mu']
        self.rho = params['rho']
        self.S = params['S'].copy()
        self.h = params['h']
        self.p = params['p']
        self.T = params['T']
        self.demand_sampler = params['demand_sampler']
        self.safety_factor = safety
        self.c = params['c'] # Base cost, defaults to 1.0

        # --- Piecewise Cost  ---
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


    def _get_marginal_cost(self, i, j, current_qty):
        """
        Compute the marginal cost to ship the *next* (qty+1)th unit from i->j.
        """
        if self.rho[i][j] == 0:
            return 0.0

        # Find which segment the (current_qty + 1) falls into
        # We look for the segment where tc_u[k] <= current_qty < tc_u[k+1]
        marginal = self.tc_m[-1] # Default to the last marginal cost
        for k in range(self.K):
            seg_start = self.tc_u[k]
            seg_end = self.tc_u[k+1] if (k+1) < len(self.tc_u) else float('inf')
            
            # The next unit is the (current_qty + 1)-th unit.
            # Since breakpoints are start-inclusive, we check if current_qty is
            # less than the end of the segment.
            if current_qty < seg_end:
                marginal = self.tc_m[k]
                break
        
        # The base cost 'c' is now part of the marginals in the DP
        # but here we apply it for consistency.
        return self.c * self.rho[i][j] * marginal


    def _compute_TIE(self, x, d):
        """
        Compute a Transshipment Inventory Equalization (TIE) matrix z.
        This version greedily selects transshipments based on the lowest
        current marginal cost, which is necessary for concave cost functions.
        """
        inv = np.array(x, dtype=float)
        sold = np.minimum(inv, d)
        inv -= sold

        z = np.zeros((self.L, self.L), dtype=int)
        
        # Calculate needs and excesses based on runout time equalization
        mu_arr = np.array(self.mu)
        # Avoid division by zero for locations with zero average demand
        runout = np.divide(inv, mu_arr, out=np.full_like(inv, np.inf), where=mu_arr!=0)
        
        if mu_arr.sum() > 0:
            eq_runout = inv.sum() / mu_arr.sum()
        else:
            eq_runout = np.inf

        needs = np.maximum((eq_runout - runout) * mu_arr, 0).astype(int)
        excess = np.maximum((runout - eq_runout) * mu_arr, 0).astype(int)

        # Iteratively fulfill needs by finding the cheapest marginal transshipment
        while np.sum(needs) > 0 and np.sum(excess) > 0:
            best_cost = float('inf')
            best_donor = -1
            best_needy = -1

            # Find the best (donor, needy) pair based on current marginal cost
            donor_indices = np.where(excess > 0)[0]
            needy_indices = np.where(needs > 0)[0]

            for k in donor_indices:
                for j in needy_indices:
                    # Cost to ship the *next* unit from k to j
                    cost = self._get_marginal_cost(k, j, z[k, j])
                    if cost < best_cost:
                        best_cost = cost
                        best_donor = k
                        best_needy = j
            
            # If no valid transshipment can be found, break
            if best_donor == -1:
                break

            # Execute the single best transshipment
            qty_to_ship = 1 # We move one unit at a time in this greedy approach
            
            z[best_donor, best_needy] += qty_to_ship
            excess[best_donor] -= qty_to_ship
            needs[best_needy] -= qty_to_ship

        return z

    def __call__(self, x, t, d):
        """
        When the policy is invoked, compute the TIE transshipment plan.
        """
        return self._compute_TIE(x, d)

