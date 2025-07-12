from policies.policy import Policy

class NTPolicy(Policy):
    def __init__(self, params):
        """
        No Transshipment policy - keeps all inventory at original locations
        Args:
            params (dict): Must contain:
                - L (int): Number of locations
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
        Execute no-transshipment policy
        Args:
            x (list): Current inventory state (x[i] = stock at location i)
        Returns:
            list[list]: Transshipment matrix z where z[j][i] = units moved from j to i
        """
        z = [[0] * self.L for _ in range(self.L)]
        
        # Keep all inventory at original locations (diagonal entries)
        for i in range(self.L):
            z[i][i] = x[i]
            
        return z