from policies.policy import Policy


class RCPolicy(Policy):
    def __init__(self, params):
        """
        Reactive Closest (RC) transshipment policy
        Args:
            params (dict): Contains:
                - L (int): Number of locations
                - rho (list[list]): Distance/cost matrix (rho[j][i] = cost from j to i)
        """
        self.L = params['L']
        self.S = params['S'].copy()
        self.h = params['h'].copy()
        self.p = params['p'].copy()
        self.c = params['c']
        self.T = params['T']
        self.rho = params['rho']
        
    def __call__(self, x, t, d):
        """
        Execute RC policy for given inventory state x
        Args:
            x (list): Current inventory state (x[i] = stock at location i)
        Returns:
            list[list]: Transshipment matrix z where z[j][i] = units moved from j to i
        """
        z = [[0]*self.L for _ in range(self.L)]
        
        # Phase 1: Self-ship all inventory first
        for i in range(self.L):
            z[i][i] = x[i]  # Keep all stock at original location
            
        # Phase 2: Reactive redistribution for empty locations
        for i in range(self.L):
            if x[i] == 0:
                # Find candidate locations with positive stock
                candidates = [j for j in range(self.L) if x[j] > 0]
                
                if candidates:
                    # Find nearest candidate using distance matrix
                    j0 = min(candidates, key=lambda j: self.rho[j][i])
                    
                    # Move one unit from j0's self-stock to i
                    z[j0][j0] -= 1  # Remove from self-ship
                    z[j0][i] += 1    # Add to transshipment
                    
        return z
