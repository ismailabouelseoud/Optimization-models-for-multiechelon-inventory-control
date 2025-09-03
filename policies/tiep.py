import numpy as np
from policies.policy import Policy

class TIEPolicy(Policy):
    def __init__(self, params, safety=0):
        # Initialize TIEPolicy with parameters and optional safety factor
        # L: number of locations
        self.L = params['L']
        # mu: vector of average demands per period for each location
        self.mu = params['mu']
        # c: cost per unit distance for transshipment
        self.c = params['c']  # Cost per unit distance
        # rho: distance matrix between locations
        self.rho = params['rho']  # Distance matrix (L x L)
        # S: inventory levels or target stock levels (deep copy)
        self.S = params['S'].copy()
        # h: holding cost vector per period for each location
        self.h = params['h']
        # p: penalty cost vector (e.g., backorder cost) for unmet demand
        self.p = params['p']
        # T: time horizon or planning horizon (number of periods)
        self.T = params['T']
        # demand_sampler: function or generator to sample random demand
        self.demand_sampler = params['demand_sampler']
        # safety_factor: scaling factor to adjust safety stock thresholds
        self.safety_factor = safety  # Configurable safety stock parameter

    def _compute_TIE(self, x, d):
        """
        Compute a Transshipment Inventory Equalization (TIE) matrix z,
        indicating quantity to transship from donor to needy locations.
        x: current inventory levels (length L)
        d: realized demand vector (length L)
        Returns:
            z: L x L matrix of transshipment quantities
        """
        # Make a copy of inventory vector to avoid modifying input
        inv = np.array(x.copy())

        # Determine units sold (fulfilled demand) = min(inventory, demand)
        sold = np.minimum(inv, inv)
        # Update inventory after sales
        inv -= sold

        # Initialize transshipment matrix to zeros
        z = np.zeros((self.L, self.L), dtype=int)

        # Calculate dynamic thresholds per location (with safety factor)
        threshold = (1 + self.safety_factor) * np.array(self.mu)

        # Check if any inventory falls below its threshold
        if np.any(inv < threshold):
            # Compute runout times: how many periods until stock depletes
            runout = inv / np.array(self.mu)
            # Compute equivalent runout if total inventory pooled and distributed
            eq_runout = inv.sum() / np.array(self.mu).sum()

            # Compute 'needs' for each location: amount to reach equalized runout
            needs = np.maximum((eq_runout - runout) * self.mu, 0)
            # Compute 'excess' stock at locations above the equalized runout
            excess = np.maximum((runout - eq_runout) * self.mu, 0)

            # Identify indices of needy and donor locations
            needy_indices = np.where(needs > 0)[0]
            donor_indices = np.where(excess > 0)[0]

            # Sort needy locations by descending need severity (largest need first)
            needy_order = np.argsort(-needs[needy_indices])

            # Allocate transshipments from donors to each needy location
            for idx in needy_order:
                j = needy_indices[idx]
                qty_needed = needs[j]

                # Compute actual transshipment cost from each donor k to needy j
                costs = [self.c * self.rho[k][j] for k in donor_indices]
                # Sort donors by increasing cost to minimize cost
                donor_order = np.argsort(costs)

                # Iterate donors in order of cheapest cost
                idx_to_deleted = []  # track exhausted donors
                for k_idx in donor_order:
                    k = donor_indices[k_idx]
                    if qty_needed <= 0:
                        break  # fulfilled the need

                    available = excess[k]
                    # Determine quantity to ship (min of need and available)
                    qty = min(qty_needed, available)

                    if qty > 0:
                        # Record transshipment from k -> j
                        z[k, j] = qty
                        # Deduct shipped quantity from donor's excess
                        excess[k] -= qty
                        # Reduce remaining need
                        qty_needed -= qty

                        # If donor's excess exhausted, mark for removal
                        if excess[k] == 0:
                            idx_to_deleted.append(donor_indices[k_idx])

                # Remove donors that have no more excess capacity
                for idx in idx_to_deleted:
                    indices_to_delete = np.where(donor_indices == idx)
                    donor_indices = np.delete(donor_indices, indices_to_delete)

        return z

    def __call__(self, x, t, d):
        """
        When the policy is invoked, compute the TIE transshipment plan.
        x: current inventory state
        t: current time period (unused here but part of Policy interface)
        d: observed demand vector
        """
        return self._compute_TIE(x, d)
