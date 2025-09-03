import numpy as np
import itertools
from policies.policy import Policy
import math

class DPGenPolicy_C(Policy):
    """
    Dynamic Programming Policy for a multi-location inventory system
    with piecewise (concave) transshipment costs.  The piecewise cost is
    described by cumulative breakpoints `tc_u` and per-segment marginal
    costs `tc_m`.  Any quantity above the last breakpoint is charged using
    the last marginal (auto-extended).
    """

    def __init__(self, params):
        """
        params keys:
            - 'L': number of locations (e.g. 2)
            - 'S': initial inventory per location (list)
            - 'T': horizon (int)
            - 'h': holding costs list
            - 'p': prices list
            - 'c': base multiplier for distance (float)
            - 'rho': LxL distance matrix
            - 'full_demand_matrix': dict mapping demand-tuples -> probability
            - 'tc_u' or 'trans_cost_breakpoints': list of cumulative breakpoints (start at 0)
            - 'tc_m' or 'trans_cost_marginals': marginals per segment (length = len(tc_u)-1)
        """
        self.L = params['L']
        self.S = params['S']
        self.T = params['T']
        self.h = params['h']
        self.p = params['p']
        self.c = params['c']
        self.rho = params['rho']
        self.demand_pmf = params['full_demand_matrix']
        raw_tc_u = params['tc_u']
        raw_tc_m = params['tc_m']
        # Accept either name for piecewise cost inputs
        #raw_tc_u = params.get('tc_u', params.get('trans_cost_breakpoints',  )
        #raw_tc_m = params.get('tc_m', params.get('trans_cost_marginals', None))

        # Defaults: simple linear cost if not provided
        if raw_tc_u is None or raw_tc_m is None:
            # single segment, unlimited, marginal = 1 (scaled by c*rho)
            raw_tc_u = [0.0]
            raw_tc_m = [1.0]

        # Normalize breakpoints to be cumulative breakpoints starting at 0
        tc_u = list(raw_tc_u)
        if len(tc_u) == 0:
            tc_u = [0.0]
        if tc_u[0] != 0.0:
            tc_u = [0.0] + tc_u

        # If user provided segment widths instead of breakpoints, they'd likely
        # give same length as marginals; we assume they supplied breakpoints.
        # Ensure tc_u is strictly non-decreasing
        for i in range(1, len(tc_u)):
            if tc_u[i] < tc_u[i-1]:
                raise ValueError("tc_u must be non-decreasing cumulative breakpoints.")

        tc_m = list(raw_tc_m)
        # If tc_m length is one less than tc_u length it's consistent (K segments)
        # If shorter, extend last marginal; if longer, trim
        expected_K = max(1, len(tc_u) - 1)
        if len(tc_m) < expected_K:
            if len(tc_m) == 0:
                tc_m = [1.0] * expected_K
            else:
                # repeat last marginal to reach expected_K
                tc_m = tc_m + [tc_m[-1]] * (expected_K - len(tc_m))
        elif len(tc_m) > expected_K:
            tc_m = tc_m[:expected_K]

        # store normalized parameters
        self.tc_u = tc_u            # cumulative breakpoints, len = K+1
        self.tc_m = tc_m            # per-segment marginal, len = K
        self.K = len(self.tc_m)

        # Validate concavity: non-increasing marginals (allow small eps)
        for a, b in zip(self.tc_m, self.tc_m[1:]):
            if a < b - 1e-12:
                raise ValueError("tc_m must be non-increasing (decreasing marginal cost).")

        # Prepare DP tables (works when L is small; here code assumes 2 but generalizes)
        max_inv = sum(self.S)
        # V[t] is an (max_inv+1)^L table; for simplicity keep 2D for L=2 as before
        # If L>2 you'd need a higher-dim structure; we keep same shape as your original
        self.V = [ np.zeros((max_inv + 1, max_inv + 1)) for _ in range(self.T + 1) ]
        self.best_action = [
            [[None] * (max_inv + 1) for _ in range(max_inv + 1)]
            for _ in range(self.T)
        ]
        self._compute_dp_gen()

    def _piecewise_cost(self, i, j, qty):
        """
        Compute the transshipment cost to ship `qty` units from i->j.
        Uses cumulative breakpoints self.tc_u and segment marginals self.tc_m.
        Any units above last breakpoint are charged with last marginal (auto-extend).
        The returned cost already includes scaling by self.c * self.rho[i][j].
        """
        if qty <= 0:
            return 0.0

        # distance scaling
        dist_scale = self.rho[i][j] 

        remaining = qty
        total = 0.0
        # number of segments K = len(tc_m)
        for k in range(self.K):
            seg_start = self.tc_u[k]
            # segment end is next breakpoint if exists, else +inf
            seg_end = self.tc_u[k+1] if (k+1) < len(self.tc_u) else float('inf')
            seg_cap = seg_end - seg_start if math.isfinite(seg_end) else float('inf')

            take = min(remaining, seg_cap)
            if take > 0:
                total += take * self.tc_m[k] * self.c*dist_scale
                remaining -= take
            if remaining <= 0:
                break

        # If remaining > 0 (because user provided finite last breakpoint), charge last marginal
        if remaining > 0:
            total += remaining * self.tc_m[-1]* self.c* dist_scale
        return total
    def _compute_dp_gen(self):
        """
        Backwards DP over horizon; enumerates feasible split of each origin's inventory.
        (This enumeration explodes quickly with L and inventory size — intended for small L.)
        """
        def all_splits(n, k):
            # This helper function is unchanged
            for dividers in itertools.combinations(range(n + k - 1), k - 1):
                result = []
                last = 0
                for divider in dividers:
                    result.append(divider - last)
                    last = divider + 1
                result.append(n + k - 1 - last)
                yield tuple(result)

        max_inv = sum(self.S)

        for t in reversed(range(self.T)):
            # Instead of iterating based on initial inventory S, we iterate through all
            # possible inventory distributions up to the total system inventory.
            for x0 in range(max_inv + 1):
                for x1 in range(max_inv + 1 - x0):
                    # The current state `x` is a tuple (x0, x1)
                    x = (x0, x1)
            
                    best_val = -np.inf
                    best_z = None

                    # Precompute partitions for each origin's inventory in the current state `x`
                    all_z_rows = [list(all_splits(x[i], self.L)) for i in range(self.L)]

                    for z_tuple in itertools.product(*all_z_rows):
                        # z is list of lists: z[i][j]
                        z = [list(row) for row in z_tuple]

                        # post-transshipment inventory
                        y = []
                        for i in range(self.L):
                            inflow = sum(z[j][i] for j in range(self.L))
                            outflow = sum(z[i][j] for j in range(self.L))
                            y.append(x[i] + inflow - outflow)

                        # piecewise transshipment cost
                        cost = 0.0
                        for i in range(self.L):
                            for j in range(self.L):
                                if z[i][j] > 0:
                                    cost += self._piecewise_cost(i, j, z[i][j])

                        # expected value over demand pmf
                        ev = 0.0
                        for d, prob in self.demand_pmf.items():
                            reward = 0.0
                            x_post = [0] * self.L
                            for i in range(self.L):
                                reward += (
                                    self.p[i] * min(y[i], d[i])
                                    - self.h[i] * max(y[i] - d[i], 0)
                                )
                                x_post[i] = min(max(y[i] - d[i], 0), max_inv)
                            
                            # The Bellman equation and value look-up are correct
                            ev += prob * (reward + self.V[t + 1][x_post[0]][x_post[1]])

                        val = ev - cost

                        if val > best_val:
                            best_val = val
                            best_z = z

                    # Store result using the state tuple x = (x0, x1)
                    self.V[t][x[0]][x[1]] = best_val
                    self.best_action[t][x[0]][x[1]] = best_z


    def __call__(self, x, t, d):
        """
        Return precomputed optimal transshipment plan for state x at period t.
        x is expected to be a length-L inventory vector (we index [0],[1] for storage).
        """
        return self.best_action[t][x[0]][x[1]]
