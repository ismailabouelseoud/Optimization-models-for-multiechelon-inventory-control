import numpy as np
import random
import pulp
from itertools import product
import networkx as nx
from policies.policy import Policy
# -----------------------------------------------
# 2. ADP POLICY (FORWARD ADP + CAVE + NETWORK FLOW)
# -----------------------------------------------

class ForwardADP(Policy):

    @staticmethod
    def compute_shadow_prices(z, c, rho, x):
        """
        z     : current flow matrix z[i][j]
        c     : unit cost scalar
        rho   : distance matrix rho[i][j]
        x     : current inventory levels x[i] (upper bounds)
        Returns: (pi_plus, pi_minus) lists of length L
        """
        L = len(x)
        G = nx.DiGraph()
        sink = 's'
        # build residual graph
        for i in range(L):
            G.add_edge(i, sink, weight=0)  # supply arc to sink
            for j in range(L):
                cost = c * rho[i][j]
                if z[i][j] < x[i]:
                    G.add_edge(i, j, weight=cost)   # forward edge (positive cost)
                if z[i][j] > 0:
                    G.add_edge(j, i, weight=-cost)  # backward edge (negative cost)
        
        pi_plus = [float('inf')] * L
        pi_minus = [float('inf')] * L

        # Compute π⁺ using Bellman-Ford
        for i in range(L):
            try:
                # Bellman-Ford for shortest path from i to sink
                length = nx.single_source_bellman_ford_path_length(G, i, weight='weight')
                pi_plus[i] = length.get(sink, float('inf'))
            except nx.NetworkXUnbounded:
                pi_plus[i] = float('-inf')  # negative cycle detected

        # Compute π⁻ using reversed graph (Bellman-Ford again)
        G_rev = G.reverse()
        for i in range(L):
            try:
                length = nx.single_source_bellman_ford_path_length(G_rev, i, weight='weight')
                pi_minus[i] = length.get(sink, float('inf'))
            except nx.NetworkXUnbounded:
                pi_minus[i] = float('-inf') 

        return pi_plus, pi_minus


    def __init__(self, params,
                 breakpoints=[0,10,20], iterations=1000, delta=2,
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
        
        self._learn()

    def _network_flow_decision(self, x):
        """Solve network flow LP for current state"""
        model = pulp.LpProblem("NetworkFlow", pulp.LpMaximize)
        
        # Decision variables
        z = pulp.LpVariable.dicts("z", 
            ((i,j) for i in range(self.L) for j in range(self.L)),
            lowBound=0, cat='Integer')
        
        g = {}
        for i in range(self.L):
            for k in range(len(self.u[i])):
                ub = (self.u[i][k+1] - self.u[i][k] 
                      if k+1 < len(self.u[i]) else None)
                g[(i,k)] = pulp.LpVariable(f"g_{i}_{k}", 
                    lowBound=0, upBound=ub, cat='Continuous')
        
        # Objective function
        obj = pulp.lpSum(
            -self.c * self.rho[i][j] * z[i,j]
            for i,j in product(range(self.L), repeat=2)
        ) + pulp.lpSum(
            self.v[i][k] * g[(i,k)]
            for i in range(self.L) for k in range(len(self.u[i]))
        )
        model += obj
        
        # Constraints
        for i in range(self.L):
            model += pulp.lpSum(z[i,j] for j in range(self.L)) == x[i]
            model += pulp.lpSum(z[j,i] for j in range(self.L)) == \
                     pulp.lpSum(g[(i,k)] for k in range(len(self.u[i])))
        
        model.solve(pulp.PULP_CBC_CMD(msg=0))
        
        return [[int(z[i,j].value()) for j in range(self.L)] 
                for i in range(self.L)]

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
                pi_sp_plus, pi_sp_minus = ForwardADP.compute_shadow_prices(z, self.c, self.rho, x)
                
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

