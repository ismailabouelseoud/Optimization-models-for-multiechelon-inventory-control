import math
import pulp
from itertools import product
from policies.policy import Policy

class LPPolicyMILP(Policy):
    """
    MILP policy: exact piecewise concave transshipment cost modeled with segment
    flows s_{i,j,t,k} and binary activations y_{i,j,t,k} to enforce ordering.
    """

    def __init__(self, params):
        self.L = params['L']
        self.S = params['S'].copy()
        self.h = params['h'].copy()
        self.p = params['p'].copy()
        self.c = params['c']
        self.T = params['T']
        self.rho = params['rho']

        # piecewise cost inputs (global for all arcs)
        raw_tc_u = params.get('tc_u', None)                # cumulative breakpoints
        raw_tc_m = params.get('tc_m', None)                # marginals per segment

        # normalize defaults: if not provided, revert to linear single segment
        if raw_tc_u is None or raw_tc_m is None:
            # single segment unlimited with marginal = 1 (scaled by c*rho)
            raw_tc_u = [0.0]
            raw_tc_m = [1.0]

        # ensure breakpoints start with 0
        tc_u = list(raw_tc_u)
        if len(tc_u) == 0 or tc_u[0] != 0.0:
            tc_u = [0.0] + tc_u if len(tc_u) else [0.0]

        # number of segments K is based on length of tc_m
        tc_m = list(raw_tc_m) if raw_tc_m is not None else [1.0]
        self.K = len(tc_m)
        self.tc_u = tc_u
        self.tc_m = tc_m

        # compute segment widths (numerical). For segments beyond breakpoints, use bigM.
        # safe bigM: max total inventory available across system (sum of initial S)
        self.bigM = sum(self.S)
        self.seg_widths = []
        for k in range(self.K):
            if k < len(self.tc_u) - 1:
                width = self.tc_u[k+1] - self.tc_u[k]
                # protect against negative widths
                if width < 0:
                    raise ValueError("tc_u must be non-decreasing cumulative breakpoints.")
                self.seg_widths.append(width)
            else:
                self.seg_widths.append(self.bigM)  # beyond last breakpoint: use bigM

    def _solve_multi_period_milp(self, x, d_callable):
        """
        Solve the exact MILP over the whole horizon (0..T-1).
        x: initial inventory vector (length L)
        d_callable: function d(t) -> demand vector at period t
        Returns: dictionary solution[t] = LxL matrix of shipments (ints)
        """
        model = pulp.LpProblem("MultiPeriod_Transshipment_MILP", pulp.LpMaximize)
        L = self.L
        T = self.T
        K = self.K

        idx_i = range(L); idx_j = range(L); idx_t = range(T); idx_k = range(K)

        # decision variables
        # zd, zs: integer shipped units (as in your earlier model)
        zd = pulp.LpVariable.dicts("zd", ((i, j, t) for i, j, t in product(idx_i, idx_j, idx_t)),
                                   lowBound=0, cat='Integer')
        zs = pulp.LpVariable.dicts("zs", ((i, j, t) for i, j, t in product(idx_i, idx_j, idx_t)),
                                   lowBound=0, cat='Integer')

        # per-segment flow s_{i,j,t,k} (integer) and binary y_{i,j,t,k}
        s = pulp.LpVariable.dicts("s", ((i, j, t, k) for i, j, t, k in product(idx_i, idx_j, idx_t, idx_k)),
                                  lowBound=0, cat='Integer')
        y = pulp.LpVariable.dicts("y", ((i, j, t, k) for i, j, t, k in product(idx_i, idx_j, idx_t, idx_k)),
                                  cat='Binary')

        # Objective: maximize profit over all periods
        # profit = sum_t sum_{i,j} ( p_j * zd - h_j * zs - c * rho_ij * sum_k m_k * s_{ijkt} )
        obj_terms = []
        for i, j, t, k in product(idx_i, idx_j, idx_t, idx_k):
            obj_terms.append(- self.c * self.rho[i][j] * self.tc_m[k] * s[(i, j, t, k)])
        for i, j, t in product(idx_i, idx_j, idx_t):
            obj_terms.append(self.p[j] * zd[(i, j, t)])
            obj_terms.append(- self.h[j] * zs[(i, j, t)])
        model += pulp.lpSum(obj_terms)

        # Link segment flows to total shipped on each arc and period:
        # sum_k s_{i,j,t,k} == zd_{i,j,t} + zs_{i,j,t}
        for i, j, t in product(idx_i, idx_j, idx_t):
            model += pulp.lpSum(s[(i, j, t, k)] for k in idx_k) == zd[(i, j, t)] + zs[(i, j, t)], \
                     f"link_sum_s_{i}_{j}_{t}"

        # Segment capacity via activation: s_{...k} <= width_k * y_{...k}
        for i, j, t, k in product(idx_i, idx_j, idx_t, idx_k):
            width = self.seg_widths[k]
            model += s[(i, j, t, k)] <= width * y[(i, j, t, k)], f"seg_cap_{i}_{j}_{t}_{k}"

        # Monotonicity of y: y_k >= y_{k+1}
        for i, j, t in product(idx_i, idx_j, idx_t):
            for k in range(K - 1):
                model += y[(i, j, t, k)] >= y[(i, j, t, k + 1)], f"y_monot_{i}_{j}_{t}_{k}"

        # Initial inventory constraint (period 0):
        for i in idx_i:
            model += pulp.lpSum(zd[(i, j, 0)] + zs[(i, j, 0)] for j in idx_j) == x[i], f"init_inv_{i}"

        # Inventory balance for t >= 1:
        for t in range(1, T):
            for i in idx_i:
                model += pulp.lpSum(zd[(i, j, t)] + zs[(i, j, t)] for j in idx_j) == \
                         pulp.lpSum(zs[(j, i, t - 1)] for j in idx_j), f"inv_balance_{i}_{t}"

        # Demand constraints: sum_i zd_{i,j,t} <= demand(t)[j]
        for t in idx_t:
            dvec = d_callable(t)
            for j in idx_j:
                model += pulp.lpSum(zd[(i, j, t)] for i in idx_i) <= dvec[j], f"demand_{j}_{t}"

        # Solve MILP
        solver = pulp.PULP_CBC_CMD(msg=0)
        model.solve(solver)

        # Check solution status
        if pulp.LpStatus[model.status] != 'Optimal':
            print(f"Warning: MILP solution status is {pulp.LpStatus[model.status]}, may not be optimal.")

        # Extract integer solution matrices per period
        solution = {}
        for t in idx_t:
            matrix = []
            for i in idx_i:
                row = []
                for j in idx_j:
                    zd_val = pulp.value(zd[(i, j, t)])
                    zs_val = pulp.value(zs[(i, j, t)])
                    if zd_val is None or zs_val is None:
                        # If variable value is None, use 0
                        total_val = 0
                    else:
                        total_val = int(round(zd_val + zs_val))
                    row.append(total_val)
                matrix.append(row)
            solution[t] = matrix

        return solution

    def __call__(self, x, t, d):
        """
        Solve the whole horizon MILP and return full solution dict (period->matrix).
        For interface compatibility you may pick solution[t] or the decision for current period.
        """
        sol = self._solve_multi_period_milp(x, d)
        return sol