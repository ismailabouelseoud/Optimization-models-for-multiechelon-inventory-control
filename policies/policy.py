
class Policy:
    def __init__(self, params):
        pass

    def __str__(self):
        msg = f"Policy: {self.__class__.__name__}, Values: \n"
        for key, value in self.__dict__.items():
            if key == "demand_sampler":
                value = value(self.T-1)
            msg += f"  {key}: {value}\n"
        return msg

    def __call__(self, x, t, d):
        pass

    def get_sim_results(self, params, x, z, d,CC):
        L = params['L']

        # --------------------------
        # Calculate transshipment costs using global piecewise decreasing marginal cost
        # --------------------------
        def piecewise_cost(qty):
            if qty <= 0:
                return 0.0
            breakpoints = params['tc_u']      # global quantity breakpoints
            marginals   = params['tc_m']      # global marginal costs

            total_cost = 0.0
            remaining = qty

            for k, bp in enumerate(breakpoints):
                if k == len(breakpoints) - 1:
                    # Last segment: extend marginal indefinitely
                    total_cost += remaining * marginals[k]
                    return total_cost
                segment_len = breakpoints[k+1] - breakpoints[k]
                if remaining <= segment_len:
                    total_cost += remaining * marginals[k]
                    return total_cost
                else:
                    total_cost += segment_len * marginals[k]
                    remaining -= segment_len

            return total_cost
        if CC :
            cost = sum(
                piecewise_cost(z[i][j])*params['c']*params['rho'][i][j] 
                for i in range(L)
                for j in range(L)
            )
        else:
            cost = sum(params['c'] * params['rho'][i][j] * z[i][j]
                    for i in range(L) for j in range(L))        

        # --------------------------
        # Inventory update
        # --------------------------
        y = [x[i] + sum(z[j][i] - z[i][j] for j in range(L))
             for i in range(L)]

        # --------------------------
        # Reward calculation
        # --------------------------
        reward = sum(
            params['p'][i] * min(y[i], d[i]) -
            params['h'][i] * max(y[i] - d[i], 0)
            for i in range(L)
        )

        profit = reward - cost
        return profit, y

