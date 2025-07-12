import pulp
from itertools import product
from policies.policy import Policy


class LPPolicy(Policy):
    def __init__(self, params):
        """
        Real-time optimal transshipment policy
        Args:
            S (list): Initial inventory levels
            h (list): Holding costs per location
            p (list): Selling prices per location
            c (float): Transshipment cost per unit distance
            rho (2D list): Distance matrix
        """
        self.L = params['L']
        self.S = params['S'].copy()
        self.h = params['h'].copy()
        self.p = params['p'].copy()
        self.c = params['c']
        self.T = params['T']
        self.rho = params['rho']


    def _solve_multi_period_lp(self,x,d):
        """Solve the complete multi-period LP"""
        model = pulp.LpProblem("MultiPeriod_Transshipment", pulp.LpMaximize)
        T = self.T
        
        # Create variables for all periods
        zd = pulp.LpVariable.dicts("zd", 
            ((i,j,t) for i,j,t in product(range(self.L), range(self.L), range(T))),
            lowBound=0, cat='Integer')
        
        zs = pulp.LpVariable.dicts("zs", 
            ((i,j,t) for i,j,t in product(range(self.L), range(self.L), range(T))),
            lowBound=0, cat='Integer')
        
        # Objective function (Equation 17)
        obj = pulp.lpSum(
            -self.c * self.rho[i][j] * (zd[i,j,t] + zs[i,j,t]) +
            self.p[j] * zd[i,j,t] -
            self.h[j] * zs[i,j,t]
            for i,j,t in product(range(self.L), range(self.L), range(T))
        )
        model += obj
        
        # Constraints (Equations 18-21)
        # Initial inventory (Equation 19)
        for i in range(self.L):
            model += pulp.lpSum(zd[i,j,0] + zs[i,j,0] for j in range(self.L)) == x[i]
        
        # Inventory balance (Equation 18)
        for t in range(1, T):
            for i in range(self.L):
                model += pulp.lpSum(zd[i,j,t] + zs[i,j,t] for j in range(self.L)) == \
                         pulp.lpSum(zs[j,i,t-1] for j in range(self.L))
        
        # Demand constraints (Equation 20)
        for t in range(T):
            for j in range(self.L):
                model += pulp.lpSum(zd[i,j,t] for i in range(self.L)) <= d(t)[j]
        
        model.solve(pulp.PULP_CBC_CMD(msg=0))        
        # Return combined solution matrix
        solution = {}
        for t in range(T):
            solution[t] = [
                    [
                        int(zd[i,j,t].varValue) + int(zs[i,j,t].varValue)
                        for j in range(self.L)
                    ]
                    for i in range(self.L)
                ]
        return solution

    def __call__(self, x, t, d):
        """
        Get optimal transshipment decision
        Args:
            x (list): Current inventory levels
            t (int): Current period (for interface compatibility)
            d (list): Current demand vector
        Returns:
            list: Transshipment matrix z where z[i][j] = units from i to j
        """
        solution = self._solve_multi_period_lp(x, d)
        return solution
    
