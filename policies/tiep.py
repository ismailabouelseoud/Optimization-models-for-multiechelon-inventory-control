import numpy as np
from policies.policy import Policy


class TIEPolicy(Policy):
    def __init__(self, params, safety=0):
        self.L = params['L']
        self.mu = params['mu']
        self.c = params['c']  # Cost per unit distance
        self.rho = params['rho']  # Distance matrix
        self.S = params['S'].copy()
        self.h = params['h']
        self.p = params['p']
        self.T=params['T']
        self.demand_sampler = params['demand_sampler']
        self.safety_factor = safety  # Configurable parameter

    def _compute_TIE(self, x, d):
        inv = np.array(x.copy())

        sold = np.minimum(inv, d)
        inv -= sold
        z = np.zeros((self.L, self.L), dtype=int)
            
        # Dynamic threshold calculation
        threshold = (1 + self.safety_factor) * np.array(self.mu)
        if np.any(inv < threshold):
            runout = inv / np.array(self.mu)
            eq_runout = inv.sum() / np.array(self.mu).sum()
            needs = np.maximum((eq_runout - runout) * self.mu, 0)
            excess = np.maximum((runout - eq_runout) * self.mu, 0)
            
            needy_indices = np.where(needs > 0)[0]
            donor_indices = np.where(excess > 0)[0]

            
            # Sort by need severity
            needy_order = np.argsort(-needs[needy_indices])

            for idx in needy_order:
                j = needy_indices[idx]
                qty_needed = needs[j]
                
                # Sort donors by actual transshipment cost

                costs = [self.c * self.rho[k][j] for k in donor_indices]
                donor_order = np.argsort(costs)

                idx_to_deleted=[]
                for k_idx in donor_order:

                    k = donor_indices[k_idx]
                    if qty_needed <= 0:
                        break
                    
                    available = excess[k]
                    qty = min(qty_needed, available)
                    
                    if qty > 0:
                        z[k, j] = qty
                        excess[k] -= qty
                        qty_needed -= qty
                        
                        if excess[k] == 0:

                            idx_to_deleted.append(donor_indices[k_idx])

                for idx in idx_to_deleted:
                    indices_to_delete = np.where(donor_indices == idx)

                    donor_indices = np.delete(donor_indices, indices_to_delete)


        return z

    def __call__(self, x, t, d):
        
        return self._compute_TIE(x, d)