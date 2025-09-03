import numpy as np
from scipy.stats import poisson
from scipy.stats import nbinom
import random

def uniform_demand_pmf(range_d1, range_d2):

    """
    Builds the joint PMF of two independent discrete uniform demands.

    Args:
        range_d1 (int): Maximum demand for product_1; demand is uniform over {0,…,range_d1}.
        range_d2 (int): Maximum demand for product_2; demand is uniform over {0,…,range_d2}.

    Returns:
        dict[tuple[int,int], float]:
            Keys are (d1, d2) pairs; values are P(D1=d1 and D2=d2), with each marginal
            P(Di = k) = 1/(range_di + 1).
    """

    pmf = {}
    num_d1 = range_d1+1
    num_d2 = range_d2+1
    prob_d1 = 1 / num_d1 if num_d1 > 0 else 0
    prob_d2 = 1 / num_d2 if num_d2 > 0 else 0
    for d1 in range(range_d1+1):
        for d2 in range(range_d2+1):
            pmf[(d1, d2)] = prob_d1 * prob_d2
    return pmf

def uniform_demand_sampler(bi1,bi2):
    def sampler(t):
        # Sample d1 and d2 independently using the provided rates
        return (random.randint(0,bi1), random.randint(0,bi2))
    return sampler 

def negbinomial_demand_pmf(max_d1, max_d2, n1, p1, n2, p2):
    """
    Calculates the joint Probability Mesh Function for two independent Negative Binomial distributed demands.

    Args:
        max_d1: The maximum value of d1 (number of failures) to include in the PMF dictionary.
        max_d2: The maximum value of d2 (number of failures) to include in the PMF dictionary.
        n1: The shape parameter (number of successes) for the first Negative Binomial distribution (demand d1).
        p1: The probability of success for the first Negative Binomial distribution (demand d1).
        n2: The shape parameter (number of successes) for the second Negative Binomial distribution (demand d2).
        p2: The probability of success for the second Negative Binomial distribution (demand d2).

    Returns:
        A dictionary representing the joint PMF, where keys are tuples (d1, d2)
        and values are the probabilities P(d1=k1, d2=k2).
    """
    pmf = {}
    # Negative Binomial distribution is defined for k >= 0 failures
    for d1 in range(max_d1 + 1):
        # Calculate the probability for d1 from the first Negative Binomial distribution
        # scipy.stats.nbinom.pmf(k, n, p) gives P(k failures before n successes)
        prob_d1 = nbinom.pmf(d1, n1, p1)
        for d2 in range(max_d2 + 1):
            # Calculate the probability for d2 from the second Negative Binomial distribution
            prob_d2 = nbinom.pmf(d2, n2, p2)
            # Since d1 and d2 are independent, the joint probability is the product
            pmf[(d1, d2)] = prob_d1 * prob_d2
    return pmf


def make_negbinomial_demand_sampler(n1, p1, n2, p2):
    """
    Creates a sampler function for two independent Negative Binomial distributed demands.

    Args:
        n1: The shape parameter (number of successes) for the first Negative Binomial distribution (demand d1).
        p1: The probability of success for the first Negative Binomial distribution (demand d1).
        n2: The shape parameter (number of successes) for the second Negative Binomial distribution (demand d2).
        p2: The probability of success for the second Negative Binomial distribution (demand d2).

    Returns:
        A function that, when called, returns a tuple (d1, d2) of sampled demands (number of failures).
    """
    def sampler(t):
        # Sample d1 and d2 independently using the provided parameters n, p
        return (int(np.random.negative_binomial(n1, p1)), int(np.random.negative_binomial(n2, p2)))
    return sampler


def poisson_demand_pmf(max_d1, max_d2, rate1, rate2):
    """
    Calculates the joint PMF for two independent Poisson distributed demands.

    Args:
        max_d1: The maximum value of d1 to include in the PMF dictionary.
        max_d2: The maximum value of d2 to include in the PMF dictionary.
        rate1: The rate parameter (lambda) for the first Poisson distribution (demand d1).
        rate2: The rate parameter (lambda) for the second Poisson distribution (demand d2).

    Returns:
        A dictionary representing the joint PMF, where keys are tuples (d1, d2)
        and values are the probabilities P(d1, d2).
    """
    pmf = {}
    for d1 in range(max_d1 + 1):
        # Calculate the probability for d1 from the first Poisson distribution
        prob_d1 = poisson.pmf(d1, rate1)
        for d2 in range(max_d2 + 1):
            # Calculate the probability for d2 from the second Poisson distribution
            prob_d2 = poisson.pmf(d2, rate2)
            # Since d1 and d2 are independent, the joint probability is the product
            pmf[(d1, d2)] = prob_d1 * prob_d2
    return pmf


def make_poisson_demand_sampler(rate1, rate2):
    """
    Creates a sampler function for two independent Poisson distributed demands.

    Args:
        rate1: The rate parameter (lambda) for the first Poisson distribution (demand d1).
        rate2: The rate parameter (lambda) for the second Poisson distribution (demand d2).

    Returns:
        A function that, when called, returns a tuple (d1, d2) of sampled demands.
    """
    def sampler(t):
        # Sample d1 and d2 independently using the provided rates
        return (int(np.random.poisson(rate1)), int(np.random.poisson(rate2)))
    return sampler


def generate_distance_matrix(n_locations, seed=None):
    """Generate Euclidean distance matrix for locations on [0,100]² grid
    Args:
        n_locations (int): Number of locations to generate.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
    Returns:
        np.ndarray: A symmetric distance matrix of shape (n_locations, n_locations).
    """
    if seed is not None:
        np.random.seed(seed)
    coords = np.random.uniform(0, 100, size=(n_locations, 2))
    dx = coords[:, 0, np.newaxis] - coords[:, 0]
    dy = coords[:, 1, np.newaxis] - coords[:, 1]
    distance_matrix = np.sqrt(dx**2 + dy**2)
    np.fill_diagonal(distance_matrix, 0)
    return np.round(distance_matrix, 2)

def gererate_set1(locations, base_p, base_h, mu_daily, T, demand_realizations):
    """
    Generate parameters for the first experimental set with varying locations (L) and transshipment costs.  
    Args:
        locations (list): List of different numbers of locations to test.
        base_p (float): Base price for all locations.
        base_h (float): Base holding cost for all locations.
        mu_daily (float): Daily demand mean for Poisson distribution.
        T (int): Number of time periods.
        demand_realizations (dict): Pre-generated demand realizations for each location.
    Returns:
        list: List of parameter dictionaries for each configuration.
    """
    set_params = []
    for L in locations:
        p = [base_p] * L
        h = [base_h] * L
        rho = generate_distance_matrix(L, seed=42)
        S = [697] * L  # From equation S_i = μT + σ√T
        
        for c in [0.1, 0.5, 1.0]:
            demands = demand_realizations[L]
            sampler = lambda t, d=demands: d[:, t]  # Capture current period
            set_params.append({
                'Distrbution':'Poission',
                'L': L,
                'T': T,
                'h': h,
                'p': p,
                'mu': [mu_daily] * L,
                'c': c,
                'tc_u':[0,2,10],
                'tc_m':[1,0.5,0.25],
                'rho': rho.copy(),
                'S': S.copy(),
                'demand_sampler': sampler,
                'full_demand_matrix': demands.copy(),
                'expected_demand_matrix': np.full((L, T), 24.0, dtype=float)
            })
    return set_params

def gererate_set2(locations, base_p, base_h, mu_daily, T, demand_realizations):
    # Experiment Set 2: Varying L and initial inventory
    """
    Generate parameters for the second experimental set with varying locations (L) and initial inventory configurations.    
    Args:
        locations (list): List of different numbers of locations to test.
        base_p (float): Base price for all locations.
        base_h (float): Base holding cost for all locations.
        mu_daily (float): Daily demand mean for Poisson distribution.
        T (int): Number of time periods.
        demand_realizations (dict): Pre-generated demand realizations for each location.
    Returns:
        list: List of parameter dictionaries for each configuration.
    """
    set_params = []
    for L in locations:
        p = [base_p] * L
        h = [base_h] * L     
        rho = generate_distance_matrix(L, seed=42)   
        # Create different inventory configurations
        S_configs = [
            {'name': 'balanced_high', 'S': [697] * L},
            {'name': 'balanced_low', 'S': [680] * L},
            {'name': 'imbalanced', 'S': list(np.random.multinomial(697 * L, [1 / L] * L))} # Imbalanced inventory (multinomial sampling)
        ]

        for config in S_configs:
            demands = demand_realizations[L]
            sampler = lambda t, d=demands: d[:, t]  # Capture current period
            set_params.append({
                'Distrbution':'Poission',
                'L': L,
                'T': T,
                'h': h,
                'p': p,
                'mu': [mu_daily]*L,
                'c': 1,
                'tc_u': [0,5,20,30,40],
                'tc_m':[1, 0.5, 0.25, 0.2, 0.1],
                'rho': rho.copy(),
                'S': config['S'].copy(),
                'demand_sampler': sampler,
                'full_demand_matrix': demand_realizations[L].copy(),
                'expected_demand_matrix':  np.full((L, T), 24.0, dtype=float)
            })
    return set_params

def generate_set3(ri, rhos, pin, T, max_d):
    """
    Generate parameters for the third experimental set with two locations 
    and negative binomial demand. This version creates all combinations of 'h' and 'p'.
    
    Args:
        ri (list): List of realizations for the negative binomial distribution.
        rhos (list): List of distances between locations.
        pin (float): Probability of success in the negative binomial distribution.
        T (int): Number of time periods.
        max_d (list): Maximum demand realizations for the negative binomial distribution.
        
    Returns:
        list: List of parameter dictionaries for each configuration.
    """
    set_params = []
    # Define h and p values outside the loop for clarity
    h_values_1 = [8]
    p_values_1 = [40]
    h_values_2 = [12]
    p_values_2 = [80]

    for k in range(len(ri)):
        for r in range(len(ri)):
            i = ri[k]
            j = ri[r]

            # Calculate mean, variance, and then standard deviation (sqrt of variance)
            mu1 = i * (1 - pin) / pin
            variance1 = i * (1 - pin) / (pin**2)
            sigma1 = np.sqrt(variance1)
            
            mu2 = j * (1 - pin) / pin
            variance2 = j * (1 - pin) / (pin**2)
            sigma2 = np.sqrt(variance2)
    
            # Inventory up to a level
            S1 = int(np.floor(mu1 * T + sigma1 * np.sqrt(T)))  
            S2 = int(np.floor(mu2 * T + sigma2 * np.sqrt(T))) 

            for rho in rhos:
                # --- Loop through each h and p value to create all combinations ---
                for h_val in h_values_1:
                    for h_val_1 in h_values_2:
                        for p_val in p_values_1:
                            for p_val_1 in p_values_2:
                                params = {
                                    'Distrbution': 'NegBin',
                                    'L'   : 2,
                                    'T'   : T,
                                    'h'   : [h_val,h_val_1],  # Assign the individual h value
                                    'c'   : 1,
                                    'tc_u': [0, 2, 4, 8],
                                    'tc_m': [0.5, 0.3, 0.2, 0.1],
                                    'p'   : [p_val,p_val_1],  # Assign the individual p value
                                    'mu'  : [mu1, mu2],
                                    'rho' : [[0, int(rho)], [int(rho), 0]],
                                    'S'   : [S1, S2],
                                    "ij"  : [i, j],
                                    'demand_sampler': make_negbinomial_demand_sampler(i, pin, j, pin),
                                    'full_demand_matrix': negbinomial_demand_pmf(max_d[r], max_d[k], i, pin, j, pin),
                                    'expected_demand_matrix': np.vstack((np.full(T, mu1, dtype=float), np.full(T, mu2, dtype=float)))
                                }
                                set_params.append(params)
    return set_params

def generate_set4(poisson_lis, rhos, T, max_d):
    """
    Generate parameters for the fourth experimental set with two locations and Poisson demand.
    This version creates all combinations of 'h' and 'p'.

    Args:
        poisson_lis (list): List of Poisson rates (lambda).
        rhos (list): List of distances between locations.
        T (int): Number of time periods.
        max_d (list): Maximum demand realizations for the Poisson distribution.

    Returns:
        list: List of parameter dictionaries for each configuration.
    """
    set_params = []
    # Define h and p values outside the loop for clarity
    h_values_1 = [8]
    p_values_1 = [40]
    h_values_2 = [12]
    p_values_2 = [80]

    for k in range(len(poisson_lis)):
        for r in range(len(poisson_lis)):

            i = poisson_lis[k]
            j = poisson_lis[r]

            # Calculate mu and sigma for Poisson distribution
            # For Poisson, mean (mu) = lambda and standard deviation (sigma) = sqrt(lambda)
            mu1, sigma1 = i, np.sqrt(i)
            mu2, sigma2 = j, np.sqrt(j)

            # Inventory up to a level
            S1 = int(np.floor(mu1 * T + sigma1 * np.sqrt(T)))
            S2 = int(np.floor(mu2 * T + sigma2 * np.sqrt(T)))

            for rho in rhos:
                # ---  Loop through each h and p value to create all combinations ---
                for h_val in h_values_1:
                    for h_val_1 in h_values_2:
                        for p_val in p_values_1:
                            for p_val_1 in p_values_2:
                                params = {
                                    'Distrbution': 'Poisson',
                                    'L'   : 2,
                                    'T'   : T,
                                    'h'   : [h_val,h_val_1],  # Assign the individual h value
                                    'c'   : 1,
                                    'tc_u': [0, 2, 4, 8],
                                    'tc_m': [0.5, 0.3, 0.2, 0.1],
                                    'p'   : [p_val,p_val_1],  # Assign the individual p value
                                    'mu'  : [mu1, mu2],
                                    'rho' : [[0, int(rho)], [int(rho), 0]],
                                    'S'   : [S1, S2],
                                    "ij"  : [i, j],
                                    'demand_sampler': make_poisson_demand_sampler(i, j),
                                    'full_demand_matrix': poisson_demand_pmf(max_d[r], max_d[k], i, j),
                                    'expected_demand_matrix': np.vstack((np.full(T, mu1, dtype=float), np.full(T, mu2, dtype=float)))
                                }
                                set_params.append(params)
    return set_params
def generate_set5(uniform_lis, rhos, T):
    """
    Generate parameters for the experimental set with all combinations of h and p.
    
    Args:
        uniform_lis (list): List of uniform values [0,b].
        rhos (list): List of distances between locations.
        T (int): Number of time periods.
        
    Returns:
        list: List of parameter dictionaries for each configuration.
    """
    set_params = []
    # Define h and p values outside the loop for clarity
    h_values_1 = [8]
    p_values_1 = [40]
    h_values_2 = [12]
    p_values_2 = [80]
    for k in range(len(uniform_lis)):
        for r in range(len(uniform_lis)):
            i = uniform_lis[k]
            j = uniform_lis[r]
            
            # The standard deviation of a uniform distribution U(0, b) is b / sqrt(12)
            mu1, sigma1 = i/2, np.sqrt((i-0)/np.sqrt(12))    
            mu2, sigma2 = j/2, np.sqrt((j-0)/np.sqrt(12))
           
            #inventory up to a level eq 16 paper :
            # "Approximate dynamic programming for lateral transshipment problems in multi-location inventory systems. Joern Meissner and Olga V. Senicheva"

            S1 = int(np.floor(mu1*T + sigma1*np.sqrt(T)))  
            S2 = int(np.floor(mu2*T + sigma2*np.sqrt(T)))    

            for rho in rhos:
                #  Loop through each h and p value to create all combinations ---
                for h_val in h_values_1:
                    for h_val_1 in h_values_2:
                        for p_val in p_values_1:

                            for p_val_1 in p_values_2:
                                params = {
                                'Distrbution': 'Uniform',
                                'L'   : 2,
                                'T'   : T,
                                'h'   : [h_val,h_val_1],  # Assign the individual h value
                                'c'   : 1,
                                'tc_u': [0, 2, 4, 8],
                                'tc_m': [0.5, 0.3, 0.2, 0.1],
                                'p'   : [p_val,p_val_1],  # Assign the individual p value
                                'mu'  : [mu1, mu2],
                                'rho' : [[0, int(rho)], [int(rho), 0]],
                                'S'   : [S1, S2],
                                "ij"  : [i, j],
                                'demand_sampler': uniform_demand_sampler(i, j),
                                'full_demand_matrix': uniform_demand_pmf(i, j),
                                'expected_demand_matrix': np.vstack((np.full(T, mu1, dtype=float), np.full(T, mu2, dtype=float)))
                                }
                                set_params.append(params)
    return set_params


def generate_parameters():
    """
    Generate parameters for all experimental sets.
    Returns:
        list: List of parameter dictionaries for each experimental set.
    """
    # Common parameters
    params_list = []
    T = 28  # Time periods derived from S=697 calculation
    mu_daily = 24  # Poisson rate
    np.random.seed(42)  # Global seed for reproducibility
    base_p = 80  # Same price for all locations
    base_h = 5  

    # Pregenerate demand realizations for all locations
    locations = [5, 10, 15, 20]
    # Generate demand realizations for each location
    demand_realizations = {
        L: np.random.poisson(mu_daily, size=(L, T))
        for L in locations
    }
    
    set_params = gererate_set1(locations, base_p, base_h, mu_daily, T, demand_realizations)
    params_list.append(set_params)
    set_params = gererate_set2(locations, base_p, base_h, mu_daily, T, demand_realizations)
    params_list.append(set_params)

    # Experiment Set 3: 2 Location, negbinomial demand
    rhos = [29, 61] # Distance between locations
    pin = 0.8
    T = 4

    ri = [2, 4, 6] # Realizations of the negative binomial distribution
    max_NB = [12, 14, 16] # Maximum demand realizations for the negative binomial distribution
    #max_NB = [30, 40, 50]
    # Generate parameters for the two locations with negative binomial demand

    set_params = generate_set3(ri, rhos, pin, T, max_NB)
    params_list.append(set_params)
    
    # Experiment Set 4: 2 Location, poisson demand

    poisson_lis = [0.5, 1, 1.5] # Poisson rates
    max_P = [8, 9, 10] # Poisson maximum demand realizations
    #max_P = [30, 40, 50] # Poisson maximum demand realizations

    unif_lis=[1,2,3] #unifrom [0,b_i]
    
    # Generate parameters for the two locations with poisson demand
    set_params = generate_set4(poisson_lis, rhos, T, max_P)  
    params_list.append(set_params)
    
    # Generate parameters for the two locations with uniform demand
    set_params = generate_set5(unif_lis, rhos, T)  
    params_list.append(set_params)

    return params_list


if __name__ == "__main__": 
    # Generate parameters for both experimental sets
    params_list = generate_parameters()
    print("Generated parameters for all sets:")

    # Example usage:
    n = 5
    for p_id in range(len(params_list)):
        print(f"Set {p_id + 1}: (first {n} entries)")
        for p in params_list[p_id][:n]:
            print(f"L={p['L']}, c={p['c']}, S={p['S'][:3]}..., rho_shape={len(p['rho'])} x {len(p['rho'][0])}, ")
