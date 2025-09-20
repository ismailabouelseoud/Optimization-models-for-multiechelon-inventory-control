import numpy as np
import random
import time
import os
import argparse
import sys

#import policy files
from set_parameters import generate_parameters
from policies.fadp import ForwardADP
from policies.tiep import TIEPolicy
from policies.lap import LAPolicy
from policies.ntp import NTPolicy
from policies.lpp import LPPolicy
from policies.dp import DPPolicy
from policies.dp_gen import DPGenPolicy
from policies.rcp import RCPolicy
from policies.dp_gen_cc import DPGenPolicy_C    
from policies.fadp_cc import ForwardADP_C
from policies.tiep_cc import TIEPolicy_C
from policies.lap_cc import LAPolicy_C
from policies.lpp_cc import LPPolicyMILP

def simulate_policy_multi_location(policy_fn, params,id, N=1000,CC=False):
    """
    Simulates inventory management policy across multiple locations and periods
    
    Args:
        policy_fn (callable): Policy function (x, t) → transshipment_matrix
        params (dict): Simulation parameters containing:
            - L (int): Number of locations
            - T (int): Number of periods
            - c (float): Transshipment cost
            - rho (list): Distance matrix
            - p (list): Selling prices
            - h (list): Holding costs
            - S (list): Initial inventories
            - demand_sampler (callable): Function generating demand vector
        N (int): Number of simulation trials
        
    Returns:
        float: Average profit across all simulations
    """
    profits = np.zeros(N)
    profits_opt = np.zeros(N)
    L = params['L']
    T = params['T']

    for n in range(N):
        # Set seeds for reproducibility (Common Random Numbers)
        random.seed(n)
        np.random.seed(n)
        
        x = params['S'].copy()
        total_profit = 0
        total_profit_opt = 0
        di = params['demand_sampler']
        if id ==0:
            if CC:
                OPT_solution=LPPolicyMILP(params)
            else:
                OPT_solution=LPPolicy(params)
            z_opt=OPT_solution(x, None, di)



        x = params['S'].copy()
        if id ==0 :
            x_opt = params['S'].copy()
        for t in range(T):
            # Get transshipment decision from policy
            d = di(t) 
            z = policy_fn(x, t, d)
            
            profit, y = policy_fn.get_sim_results(params, x, z, d,CC)            
            total_profit += profit
            if id ==0:
                profit_opt, y_opt = OPT_solution.get_sim_results(params, x_opt, z_opt[t], d,CC)
                total_profit_opt += profit_opt
                x_opt = [max(y_opt[i] - d[i], 0) for i in range(L)]

            # Update inventory for next period
            x = [max(y[i] - d[i], 0) for i in range(L)]

        if id ==0 :
            profits_opt[n] = total_profit_opt
        profits[n] = total_profit
    if id ==0 :
        return profits_opt.mean(),profits.mean()
    else:
        return profits.mean()



def simulate_policy_2_location(policy_fn, params,testcase_id,id, N=1000,CC=False):

    def sample_demand(bi1,bi2):

        # Monte Carlo draw: randint(0,2) returns 0,1,2 each with equal chance :contentReference[oaicite:2]{index=2}
        d1=random.randint(0, bi1)
        d2=random.randint(0, bi2)
        return [d1,d2]
    
    def sample_poisson_demand(rate1, rate2):
        """
        Samples two independent Poisson distributed demands.

        Args:
            rate1: The rate parameter (lambda) for the first Poisson distribution (demand d1).
            rate2: The rate parameter (lambda) for the second Poisson distribution (demand d2).

        Returns:
            A list [d1, d2] containing the sampled demands.
        """
        # Sample d1 from a Poisson distribution with rate rate1
        d1 = np.random.poisson(rate1)
        # Sample d2 from a Poisson distribution with rate rate2
        d2 = np.random.poisson(rate2)
        return [int(d1), int(d2)] # Ensure results are integers
    
    def sample_negbinomial_demand(n1, n2, p1=0.8, p2=0.8):
        """
        Samples two independent Negative Binomial distributed demands.

        Args:
            n1: The shape parameter (number of successes) for the first Negative Binomial distribution (demand d1).
            p1: The probability of success for the first Negative Binomial distribution (demand d1).
            n2: The shape parameter (number of successes) for the second Negative Binomial distribution (demand d2).
            p2: The probability of success for the second Negative Binomial distribution (demand d2).

        Returns:
            A list [d1, d2] containing the sampled demands (number of failures).
        """
        # Sample d1 from a Negative Binomial distribution with parameters n1, p1
        # numpy.random.negative_binomial(n, p) gives the number of *failures*
        d1 = np.random.negative_binomial(n1, p1)
        # Sample d2 from a Negative Binomial distribution with parameters n2, p2
        d2 = np.random.negative_binomial(n2, p2)
        return [int(d1), int(d2)] # Ensure results are integers    
    hi1, hi2 = params['ij']


    profits = np.zeros(N)
    profits_opt = np.zeros(N)
    if id==0:
        if CC:
            OPT_solution = DPGenPolicy_C(params)
        else: 
            OPT_solution = DPPolicy(params)
    L = params['L']
    T = params['T']
    for n in range(N):
        random.seed(n)                     # CRN for Python RNG :contentReference[oaicite:1]{index=1}
        np.random.seed(n)                  # CRN for NumPy RNG :contentReference[oaicite:2]{index=2}
        x = params['S'].copy()
        x_opt = params['S'].copy()
        total_profit = 0
        total_profit_opt = 0
        for t in range(T):
            if testcase_id==3:
                d = sample_poisson_demand(hi1, hi2) #Monte Carlo draw: randint(0,1)
            elif testcase_id==2:
                d = sample_negbinomial_demand(hi1, hi2) #Monte Carlo draw: randint(0,1)
            else :
                d = sample_demand(hi1, hi2) 
            
            z = policy_fn(x, t, d)         # call policy with state and time
            profit, y = policy_fn.get_sim_results(params, x, z, d,CC)        
            total_profit += profit
            if id ==0 :
                z_opt = OPT_solution(x, t, d)
                profit_opt, y_opt = OPT_solution.get_sim_results(params, x_opt, z_opt, d, CC)
                total_profit_opt += profit_opt
                x_opt = [max(y_opt[i] - d[i], 0) for i in range(L)]
            
            x = [max(y[i] - d[i], 0) for i in range(L)]
            #print(f"{x} {x_opt} -- {y} {y_opt} -- {z} {z_opt} -- {profit} {profit_opt}") 
        profits[n] = total_profit
        if id ==0 :
            profits_opt[n] = total_profit_opt
    if id ==0 :        
        return profits_opt.mean(),profits.mean()
    else: 
        return profits.mean()

def simulate_policy(policy, params, testcase_id,id,N=1000,CC=False):
    """
    Simulates the given policy over N iterations and returns the average profit.
    
    Args:
        policy (callable): Policy function to simulate.
        params (dict): Parameters for the simulation.
        N (int): Number of iterations to simulate.
        
    Returns:
        tuple: Average profit for optimal policy and the given policy.
    """
    if params['L'] == 2:
        print("Simulating for 2 locations")
        return simulate_policy_2_location(policy, params,testcase_id,id, N,CC)
    else:
        print("Simulating for multiple locations")
        return simulate_policy_multi_location(policy, params, id, N, CC)


def dict_until_key(d, stop_key):
    result = {}
    for k, v in d.items():
        result[k] = v
        if k == stop_key:
            break
    return result


   
if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Simulate inventory management policies.")
    parser.add_argument('--testcase_id', type=int, default=2, help='Test case ID to run (0-4), mulitlocation 0 and 1, two location: 2,3,and 4')
    parser.add_argument('--N', type=int, default=1000, help='Number of simulation iterations (default: 1000)')
    parser.add_argument('--ncost', action='store_true', help='Use non linear cost')
    parser.add_argument('--force', action='store_true', help='Force re-run of all simulations even if results exist')
    args = parser.parse_args()
    testcase_id = args.testcase_id
    if testcase_id not in [0, 1, 2, 3, 4]:
        print("Invalid testcase_id. Please choose from 0 to 4.")
        sys.exit(1)
    print(f"Running simulation for test case ID: {testcase_id}")
    

    
    # 7 policies, 2 golden (LP, DP), 5 to compare against)
    policy_factory = {
        "RCP": RCPolicy, # Reactive Closest transshipment policy
        'NT'  : NTPolicy, # No-Transshipement policy
        'TIE' : TIEPolicy, # Transshipment with Inventory Equalization policy
        "LA": LAPolicy, # Lookahead policy
        "ADP" : ForwardADP, # Adaptive Dynamic Programming policy
        "ADPC" : ForwardADP_C, # Adaptive Dynamic Programming policy
        "LP": LPPolicy, # Linear Programming policy, used as gold standard for multiple locations
        'DP'  : DPPolicy, # Dynamic Programming policy, used as gold standard for 2 locations
        'DPGen' : DPGenPolicy,  # Generalized Dynamic Programming policy
        'DPGenC' : DPGenPolicy_C, # Generalized Dynamic Programming policy
        'TIEC' : TIEPolicy_C, # Transshipment with Inventory Equalization policy
        "LAC": LAPolicy_C, # Lookahead policy
        "LPC": LPPolicyMILP # Lookahead policy

    }
    if args.ncost:
    # Dictionary of policies to run for each test case
        id2policies = {
            0: ['ADPC','TIEC', 'LAC', 'NT', 'RCP'], # Test case 1, Multiple locations, Varying locations (L) and transshipment costs.
            1: ['ADPC','TIEC', 'LAC', 'NT', 'RCP'], # Test case 2, Multiple locations, Varying locations (L) and initial inventory.
            2: ['ADPC','TIEC', 'LAC', 'NT', 'RCP'], # Test case 3, 2 locations, and negative binomial demands.
            3: ['ADPC','TIEC', 'LAC', 'NT', 'RCP'], # Test case 4, 2 locations, and Poisson demands.
            4: ['ADPC','TIEC', 'LAC', 'NT', 'RCP'], # Test case 5, 2 locations, and uniform demands.
        }
        identifier_cc='cc' #identifier for concave cost 
    else :
        id2policies = {
            0: ['ADP', 'TIE', 'LA', 'NT', 'RCP'], # Test case 1, Multiple locations, Varying locations (L) and transshipment costs.
            1: ['ADP', 'TIE', 'LA', 'NT', 'RCP'], # Test case 2, Multiple locations, Varying locations (L) and initial inventory.
            2: ['ADP','TIE', 'LA', 'NT', 'RCP'], # Test case 3, 2 locations, and negative binomial demands.
            3: ['ADP','TIE', 'LA', 'NT', 'RCP'], # Test case 4, 2 locations, and Poisson demands.
            4: ['ADP','TIE', 'LA', 'NT', 'RCP'], # Test case 5, 2 locations, and uniform demands.
        }        
    
    # Set up scenario parameters
    set_params = generate_parameters()

    policies = id2policies[testcase_id]
    params = set_params[testcase_id]
    print(f"Start Running policies {policies} test id: {testcase_id} (Total {len(set_params)} ids)")
    
    # Create results directory and initialize policy collector
    policy_collector = {}
    results = "results"
    os.makedirs(results, exist_ok=True)
    for i in range(len(params)):
        # Create a unique result file for each test case and paramerter set
        res_path = os.path.join(results, f"testcase_{testcase_id}_set_{i}.txt")
        
        if args.force and os.path.exists(res_path):
            print(f"Removing existing result file: {res_path}")
            os.remove(res_path) 
        
        flag='run' # an identifier to check if the optimization have ran or the results already avaliabile
        if os.path.exists(res_path):
            print(f"Skipping iteration {i} for test case {testcase_id}, result file already exists: {res_path}")
            print('To rerun all please use the "--force" option when running the simulate.py code')
            flag='skip'
            continue

        msg_final = ""
        # Initialize policy collector for this set of parameters
        for name in policies:
            st = time.time()
            msg = f"Iter {i}/{len(params)}. Initiating {name} policy start."
            msg_final += msg + "\n"
            print(msg)

            policy_collector[name] = policy_factory[name](params[i])
            msg = f"Set {i}/{len(params)}. Initiating {name} policy end. Elaplsed time {time.time() - st} sec."
            print(msg)
            msg_final += msg + "\n"     
            if testcase_id <= 1:
                full_str=f"Finished Running Set {i} from test id {testcase_id} with params: {dict_until_key(params[i], 'S')} "
            else:
                full_str=f"Finished Running Set {i} from test id {testcase_id} with params: {dict_until_key(params[i], 'ij')} "     
            
            msg_final += full_str + "\n"

    
        # Simulate & report
        id=0
        for name in policies:
            st = time.time()
            policy = policy_collector[name]
            avg = simulate_policy(policy, params[i],testcase_id,id,N=args.N,CC=args.ncost)
            if id ==0 :
                msg = f"Set {i}/{len(params)}, {args.N} iterations. Average profit <OPT>: {avg[0]:.2f}, average profit <{name}>: {avg[1]:.2f}. Ratio: {avg[1]/avg[0]}. Elapsed time {time.time() - st} sec."
                OPT_value=avg[0]
            else:
                msg = f"Set {i}/{len(params)}, {args.N} iterations. Average profit <OPT>: {OPT_value:.2f}, average profit <{name}>: {avg:.2f}. Ratio: {avg/OPT_value}. Elapsed time {time.time() - st} sec."
            msg_final += msg + "\n"
            print(msg)
            id=+1

        with open(res_path, "w") as f:
            f.write(msg_final)
    if flag!='skip':
        print(f"Finished Running test id: {testcase_id} (Total {len(set_params)} ids). Results saved in {results} directory." )
