import numpy as np
import random
import pulp
from itertools import product
import networkx as nx
from policies.policy import Policy
import math
# -----------------------------------------------
# 2. ADP POLICY (FORWARD ADP + CAVE + NETWORK FLOW)
# -----------------------------------------------

class ForwardADP_C(Policy):
    @staticmethod
    def get_piecewise_cost(flow_amount, breakpoints, marginals):
        """Calculate total cost for given flow amount"""
        if flow_amount <= 0:
            return 0.0
        
        total_cost = 0.0
        remaining_flow = flow_amount
        
        for i in range(len(marginals)):
            segment_capacity = breakpoints[i+1] - breakpoints[i]
            flow_in_segment = min(remaining_flow, segment_capacity)
            total_cost += flow_in_segment * marginals[i]
            remaining_flow -= flow_in_segment
            
            if remaining_flow <= 0:
                break
        
        return total_cost
    @staticmethod
    def compute_total_piecewise_cost(z, c, rho, breakpoints, marginals):
        """Compute total cost for solution with piecewise costs"""
        total_cost = 0.0
        L = len(z)
        
        for i in range(L):
            for j in range(L):
                if z[i][j] > 0:
                    flow_cost = ForwardADP_C.get_piecewise_cost(z[i][j], breakpoints, marginals)
                    total_cost += flow_cost * c * rho[i][j]
        
        return total_cost
    @staticmethod
    def compute_shadow_prices_piecewise_milp(z, c, rho, x, trans_cost_breakpoints, 
                                        trans_cost_marginals, integer_vars, 
                                        method='linearization'):
        """
        Compute π⁺ and π⁻ for MILP with piecewise linear costs
        """
        L = len(x)
        
        if method == 'linearization':
            return ForwardADP_C.compute_shadow_prices_piecewise_linearized(
                z, c, rho, x, trans_cost_breakpoints, trans_cost_marginals, integer_vars)
        elif method == 'local_approximation':
            return ForwardADP_C.compute_shadow_prices_local_approximation(
                z, c, rho, x, trans_cost_breakpoints, trans_cost_marginals, integer_vars)
        elif method == 'discrete_enumeration':
            return ForwardADP_C.compute_shadow_prices_discrete_enumeration(
                z, c, rho, x, trans_cost_breakpoints, trans_cost_marginals, integer_vars)

    @staticmethod
    def compute_shadow_prices_piecewise_linearized(z,c, rho, x, breakpoints, marginals, integer_vars):
        """
        Build expanded residual network with segment-nodes per (i,j) arc.
        Returns approximate pi_plus, pi_minus (shortest path costs from origin to sink,
        and from origin on reversed graph respectively).
        """
        L = len(x)
        G = nx.DiGraph()
        sink = 'SINK'
        
        # Create destination->sink edges (destinations lead to sink with zero cost)
        for dest in range(L):
            G.add_edge(dest, sink, weight=0.0)
        
        # Build expanded arcs: origin -> seg0_in -> seg0_out -> seg1_in -> ... -> last_out -> dest
        K = len(marginals)
        
        # Handle two possible data structures:
        # 1. K+1 breakpoints for K segments (standard piecewise linear)
        # 2. K breakpoints with K marginals (last segment potentially unbounded)
        
        for i in range(L):
            for j in range(L):
                current_flow = z[i][j] if i < len(z) and j < len(z[i]) else 0.0
                cumulative_capacity = 0.0
                prev_out = None
                
                for seg_idx in range(K):
                    seg_start = breakpoints[seg_idx]
                    
                    # Handle segment end based on available breakpoints
                    if seg_idx + 1 < len(breakpoints):
                        seg_end = breakpoints[seg_idx + 1]
                        seg_cap = seg_end - seg_start
                    else:
                        # Last segment - use a large capacity if no next breakpoint
                        seg_cap = float('inf')  # or some large number like 1000
                        seg_end = seg_start + seg_cap
                    marginal_cost = marginals[seg_idx] * c*rho[i][j]
                    
                    node_in = f"{i}_{j}_seg{seg_idx}_in"
                    node_out = f"{i}_{j}_seg{seg_idx}_out"
                    
                    # connect origin to first segment_in
                    if seg_idx == 0:
                        G.add_edge(i, node_in, weight=0.0)
                    else:
                        # connect previous segment out to this segment in
                        G.add_edge(prev_out, node_in, weight=0.0)
                    
                    # segment arc with marginal cost (forward)
                    G.add_edge(node_in, node_out, weight=marginal_cost)
                    
                    # connect this segment_out to destination when it's the last segment
                    if seg_idx == K-1:
                        G.add_edge(node_out, j, weight=0.0)
                    
                    # Add reverse arc representing ability to undo flow currently assigned (residual)
                    flow_in_segment = max(0.0, min(current_flow - cumulative_capacity, seg_cap))
                    if flow_in_segment > 0:
                        # reverse arc cost = -marginal_cost (gives benefit to reduce flow on that segment)
                        G.add_edge(node_out, node_in, weight=-marginal_cost)
                    
                    cumulative_capacity += seg_cap
                    prev_out = node_out
        
        # Now compute shortest path lengths origin -> sink
        pi_plus = [float('inf')] * L
        for i in range(L):
            try:
                dists = nx.single_source_bellman_ford_path_length(G, i, weight='weight')
                pi_plus[i] = dists.get(sink, float('inf'))
            except nx.NetworkXUnbounded:
                pi_plus[i] = float('-inf')
        
        # For pi_minus compute on reversed graph (equivalent to cheapest way to send one unit into i)
        G_rev = G.reverse(copy=True)
        pi_minus = [float('inf')] * L
        for i in range(L):
            try:
                dists = nx.single_source_bellman_ford_path_length(G_rev, i, weight='weight')
                pi_minus[i] = dists.get(sink, float('inf'))
            except nx.NetworkXUnbounded:
                pi_minus[i] = float('-inf')
        
        return pi_plus, pi_minus
    @staticmethod

    def compute_shadow_prices_local_approximation(z, c, rho, x, breakpoints, marginals, integer_vars):
        """
        Method 2: Use local marginal costs around current solution
        """
        L = len(x)
        G = nx.DiGraph()
        sink = 's'
        
        for i in range(L):
            G.add_edge(i, sink, weight=0)
            for j in range(L):
                current_flow = z[i][j]
                
                # Use marginal cost at current flow level
                marginal_cost = ForwardADP_C.get_marginal_cost(current_flow, breakpoints, marginals) * c* rho[i][j]
                
                # Standard residual graph construction with local marginal cost
                if current_flow < x[i]:
                    G.add_edge(i, j, weight=marginal_cost)
                if current_flow > 0:
                    G.add_edge(j, i, weight=-marginal_cost)
        
        # Standard π⁺, π⁻ computation
        pi_plus = [float('inf')] * L
        pi_minus = [float('inf')] * L
        
        for i in range(L):
            try:
                length = nx.single_source_bellman_ford_path_length(G, i, weight='weight')
                pi_plus[i] = length.get(sink, float('inf'))
            except nx.NetworkXUnbounded:
                pi_plus[i] = float('-inf')
        
        G_rev = G.reverse()
        for i in range(L):
            try:
                length = nx.single_source_bellman_ford_path_length(G_rev, i, weight='weight')
                pi_minus[i] = length.get(sink, float('inf'))
            except nx.NetworkXUnbounded:
                pi_minus[i] = float('-inf')
        
        return pi_plus, pi_minus

    def compute_shadow_prices_discrete_enumeration(z, c, rho, x, breakpoints, marginals, integer_vars):
        """
        Method 3: Enumerate nearby integer solutions and compute costs
        """
        L = len(x)
        
        # Generate candidate integer solutions near current z
        candidates = ForwardADP_C.generate_nearby_integer_solutions(z, integer_vars, max_distance=1)
        
        best_pi_plus = None
        best_pi_minus = None
        best_cost = float('inf')
        
        for z_candidate in candidates:
            if ForwardADP_C.is_feasible_solution(z_candidate, x):
                # Compute total piecewise cost for this candidate
                total_cost = ForwardADP_C.compute_total_piecewise_cost(z_candidate, c, rho, breakpoints, marginals)
                
                if total_cost < best_cost:
                    # Use local approximation for this integer solution
                    pi_plus, pi_minus = ForwardADP_C.compute_shadow_prices_local_approximation(
                        z_candidate, c, rho, x, breakpoints, marginals, integer_vars)
                    best_pi_plus, best_pi_minus = pi_plus, pi_minus
                    best_cost = total_cost
        
        return best_pi_plus or [0]*L, best_pi_minus or [0]*L
    @staticmethod

    def generate_nearby_integer_solutions(z, integer_vars, max_distance=1):
        """Generate integer solutions by rounding integer variables"""
        candidates = []
        L = len(z)
        
        # Start with base solution (round all integer vars)
        base_z = [[z[i][j] for j in range(L)] for i in range(L)]
        for (i, j) in integer_vars:
            base_z[i][j] = round(z[i][j])
        candidates.append(base_z)
        
        # Generate variations by floor/ceil of integer variables
        for (i, j) in integer_vars:
            if z[i][j] != round(z[i][j]):  # Only if not already integer
                # Floor version
                floor_z = [row[:] for row in base_z]
                floor_z[i][j] = math.floor(z[i][j])
                candidates.append(floor_z)
                
                # Ceil version  
                ceil_z = [row[:] for row in base_z]
                ceil_z[i][j] = math.ceil(z[i][j])
                candidates.append(ceil_z)
        
        return candidates

    @staticmethod
    def get_marginal_cost(flow_amount, breakpoints, marginals):
        """Get marginal cost at specific flow level (safe indexing)."""
        K = len(marginals)
        # expect breakpoints length = K+1
        for idx in range(K):
            # if flow is <= upper end of segment idx
            if idx + 1 < len(breakpoints):
                if flow_amount <= breakpoints[idx + 1]:
                    return marginals[idx]
            else:
                return marginals[idx]
        return marginals[-1]
    @staticmethod
    def get_piecewise_cost(flow_amount, breakpoints, marginals):
        """Calculate total cost for given flow amount safely."""
        if flow_amount <= 0:
            return 0.0
        total_cost = 0.0
        remaining = flow_amount
        K = len(marginals)
        for k in range(K):
            seg_start = breakpoints[k]
            seg_end = breakpoints[k+1] if (k+1) < len(breakpoints) else float('inf')
            seg_cap = seg_end - seg_start
            flow_in_seg = min(remaining, seg_cap)
            total_cost += flow_in_seg * marginals[k]
            remaining -= flow_in_seg
            if remaining <= 0:
                break
        # if still remaining, use last marginal
        if remaining > 0:
            total_cost += remaining * marginals[-1]
        return total_cost
    @staticmethod
    def is_feasible_solution(z, x):
        """Check if solution satisfies capacity constraints"""
        L = len(z)
        for i in range(L):
            total_flow = sum(z[i][j] for j in range(L))
            if total_flow > x[i]:
                return False
        return True 


    def __init__(self, params, breakpoints=[0,10,20], iterations=1000, delta=2,
                a=0.1, b=0.7):
        
        """
        ADP policy for multi-location transshipment problems
        Args:
            S (list): Initial inventory levels for each location
            T (int): Number of periods
            h (list): Holding costs per location
            p (list): Selling prices per location
            c (float): Transshipment cost per unit distance
            rho (2D list): Distance matrix between locations
            demand_sampler (func): Function generating demand vector
            breakpoints (list): Initial breakpoints for CAVE
            iterations (int): Number of learning iterations
            delta (int): Neighborhood size for breakpoint expansion
            a (float): Learning rate
            b (float): Exploration rate decay
        """

        """
        params['S'], 
        params['T'], 
        params['h'], 
        params['p'],
        params['c'],
        params['rho'], 
        """

        self.S = params['S'].copy()
        self.T = params['T']
        self.h = params['h'].copy()
        self.p = params['p'].copy()
        self.c = params['c']
        self.rho = [row.copy() for row in params['rho']]
        self.demand_sampler = params['demand_sampler']
        self.L = len(self.S)  # Number of locations
        
        # ADP parameters
        self.iters = iterations
        self.delta = delta
        self.a = a
        self.b = b
        
        # Initialize CAVE approximation
        self.u = [list(breakpoints) for _ in range(self.L)]
        self.v = [[100, -50, -200] for _ in range(self.L)]
        self.tc_u = params['tc_u']
        self.tc_m = params['tc_m']      
        self._learn()
    def _network_flow_decision(self, x):
        """Solve network flow MIP for current state with piecewise concave transshipment cost.

        Requires that self.tc_u (list of breakpoints) and self.tc_m (list of marginal costs)
        are defined. Marginal costs in self.tc_m
        should be nonincreasing (decreasing or equal) so total cost is concave.
        """
        model = pulp.LpProblem("NetworkFlow_MIP", pulp.LpMaximize)

        # Decision variables
        # z: integer shipped units from i to j 
        z = pulp.LpVariable.dicts("z",
                                ((i, j) for i in range(self.L) for j in range(self.L)),
                                lowBound=0, cat='Integer')

        # g: CAVE piecewise state-value variables (
        g = {}
        for i in range(self.L):
            for k in range(len(self.u[i])):
                ub = (self.u[i][k+1] - self.u[i][k] if k+1 < len(self.u[i]) else None)
                g[(i, k)] = pulp.LpVariable(f"g_{i}_{k}",
                                            lowBound=0,
                                            upBound=ub,
                                            cat='Continuous')

        # --- New: piecewise segments for transshipment cost on each arc (i,j)
        # self.tc_u: breakpoints list, e.g. [0, 5, 20]  (must cover possible shipments)
        # self.tc_m: marginal costs for each segment (same length as tc_u)
        # we'll treat segment k width = tc_u[k+1] - tc_u[k] (last segment width None -> use x[i] as upper bound)
        K = len(self.tc_u)
        s = {}   # segment flow variables per arc
        y = {}   # binary activation to enforce ordering of segments

        for i in range(self.L):
            for j in range(self.L):
                for k in range(K):
                    # compute UB for this segment (None => we'll bound by x[i])
                    if k + 1 < K:
                        seg_ub = self.tc_u[k+1] - self.tc_u[k]
                    else:
                        # last segment: allow up to current origin inventory (safe bound)
                        seg_ub = x[i]
                    # continuous / integer choice: keep integer if z is integer
                    s[(i, j, k)] = pulp.LpVariable(f"s_{i}_{j}_{k}", lowBound=0, upBound=seg_ub, cat='Integer')
                    # create binary for all but last (we can also create for last to keep uniform)
                    y[(i, j, k)] = pulp.LpVariable(f"y_{i}_{j}_{k}", cat='Binary')

        # Objective function:
        # - (transshipment costs) + (approx value via g)
        # We assume segment marginal costs are given per *unit distance* or per unit (user chooses).
        # We'll scale by distance rho[i][j] if that was your previous convention (was c * rho[i][j]).
        # Use: cost_coeff = self.tc_m[k] * self.rho[i][j]
        trans_cost_terms = []
        for i in range(self.L):
            for j in range(self.L):
                for k in range(K):
                    cost_coeff = self.tc_m[k] * self.c *self.rho[i][j]
                    trans_cost_terms.append(- cost_coeff * s[(i, j, k)])  # negative because we maximize profit (minimize cost)

        value_terms = []
        for i in range(self.L):
            for k in range(len(self.u[i])):
                value_terms.append(self.v[i][k] * g[(i, k)])

        obj = pulp.lpSum(trans_cost_terms) + pulp.lpSum(value_terms)
        model += obj

        # Constraints

        # 1) outgoing shipments sum per origin equals available inventory
        for i in range(self.L):
            model += pulp.lpSum(z[i, j] for j in range(self.L)) == x[i]

        # 2) incoming shipments at i equals the total g for that i (CAVE structure)
        for i in range(self.L):
            model += pulp.lpSum(z[j, i] for j in range(self.L)) == pulp.lpSum(g[(i, k)] for k in range(len(self.u[i])))

        # 3) link z and s: each arc's total flow equals sum of its segment flows
        for i in range(self.L):
            for j in range(self.L):
                model += z[i, j] == pulp.lpSum(s[(i, j, k)] for k in range(K))

        # 4) enforce ordering of segments using y:
        #    s_k <= width_k * y_k  (so a segment can be used only if its y_k=1)
        #    and y_k >= y_{k+1}  (so earlier/higher-cost segments must be activated first)
        for i in range(self.L):
            for j in range(self.L):
                for k in range(K):
                    # width for this k:
                    if k + 1 < K:
                        width = self.tc_u[k+1] - self.tc_u[k]
                    else:
                        width = x[i]  # last segment up to origin inventory
                    # s <= width * y
                    model += s[(i, j, k)] <= width * y[(i, j, k)]
                    # also trivial bound: s <= width (already via upBound) but above links to y
                # monotonicity of y
                for k in range(K - 1):
                    model += y[(i, j, k)] >= y[(i, j, k + 1)]

        # Solve MIP (CBC supports binaries)
        model.solve(pulp.PULP_CBC_CMD(msg=0))

        # return integer flows z in matrix form (like before)
        return [[int(round(z[i, j].value() or 0)) for j in range(self.L)] for i in range(self.L)]


    def _learn(self):
        """Main learning loop with CAVE updates"""
        for n in range(self.iters):
            x = self.S.copy()
            alpha = self.a / (5 + self.a * n)
            
            for t in range(self.T):
                # Epsilon-greedy exploration
                epsilon = self.b ** n
                if random.random() < epsilon:
                    z = self._random_policy(x)
                else:
                    z = self._network_flow_decision(x)
                
                # Update post-transshipment inventory
                y = [x[i] + sum(z[j][i] - z[i][j] 
                     for j in range(self.L)) for i in range(self.L)]
                
                # Get demand realization
                d = self.demand_sampler(t)  # Pass current period
                
                # Compute shadow prices
                #pi_sp_plus, pi_sp_minus = ForwardADP.compute_shadow_prices_rounded(z, self.c, self.rho, x)
                integer_vars = {(i, j) for i in range(self.L) for j in range(self.L)}

                # Compute π⁺ and π⁻ with piecewise costs
                pi_sp_plus, pi_sp_minus = ForwardADP_C.compute_shadow_prices_piecewise_milp(
                    z, self.c, self.rho, x, self.tc_u, self.tc_m, 
                    integer_vars, method='local_approximation'
                )                
                # Compute immediate rewards
                
                pi_r = [self.p[i] if y[i] < d[i] else -self.h[i]
                        for i in range(self.L)]
                
                # Combine price signals
                pi_plus = [pi_r[i] + (pi_sp_plus[i] if np.isfinite(pi_sp_plus[i]) else 0)
                          for i in range(self.L)]
                pi_minus = [pi_r[i] + (pi_sp_minus[i] if np.isfinite(pi_sp_minus[i]) else 0)
                           for i in range(self.L)]
                
                # Update value function approximation
                self._update_cave(pi_plus, pi_minus, x, y, alpha)
                
                # Transition to next state
                x = [max(y[i] - d[i], 0) for i in range(self.L)]

    def _random_policy(self, x):
        """Generate random feasible transshipment"""
        z = [[0]*self.L for _ in range(self.L)]
        for i in range(self.L):
            remaining = x[i]
            for j in random.sample(range(self.L), self.L):
                alloc = random.randint(0, remaining)
                z[i][j] = alloc
                remaining -= alloc
                if remaining <= 0:
                    break
        return z

    def _update_cave(self, pi_plus, pi_minus, x, y, alpha):
        """Update CAVE approximation parameters"""
        for i in range(self.L):
            # Find relevant segments
            try:
                k_minus = max(k for k in range(len(self.u[i]))
                           if self.u[i][k] <= x[i])
            except ValueError:
                k_minus = 0
                
            try:
                k_plus = min(k for k in range(len(self.u[i]))
                          if self.u[i][k] >= y[i])
            except ValueError:
                k_plus = len(self.u[i]) - 1
                
            # Update value estimates
            if k_minus < len(self.v[i]):
                self.v[i][k_minus] = alpha*pi_minus[i] + (1-alpha)*self.v[i][k_minus]
            if k_plus < len(self.v[i]):
                self.v[i][k_plus] = alpha*pi_plus[i] + (1-alpha)*self.v[i][k_plus]
                
            # Expand breakpoints
            for bp in [x[i]-self.delta, x[i]+self.delta]:
                if bp >= 0 and bp not in self.u[i]:
                    self.u[i].append(bp)
            self.u[i].sort()
            
            # Maintain parameter vectors
            while len(self.v[i]) < len(self.u[i]):
                self.v[i].append(0)

    def __call__(self, x, t, d):
        """Execute policy for current state and period"""
        return self._network_flow_decision(x)

