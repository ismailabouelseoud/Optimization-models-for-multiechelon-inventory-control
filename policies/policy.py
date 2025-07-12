
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

    def get_sim_results(self, params, x, z, d):
        #x = params['S'].copy()
        L = params['L']
        # Calculate transshipment costs
        cost = sum(params['c'] * params['rho'][i][j] * z[i][j]
                    for i in range(L) for j in range(L))

        # Generate demand vector for all locations
        # Update inventory positions
        y = [x[i] + sum(z[j][i] - z[i][j] for j in range(L))
                for i in range(L)]

        # Calculate period reward
        reward = sum(
            params['p'][i] * min(y[i], d[i]) - 
            params['h'][i] * max(y[i] - d[i], 0)
            for i in range(L)
        )
        
        # Calculate profit
        profit = reward - cost
        return profit, y
