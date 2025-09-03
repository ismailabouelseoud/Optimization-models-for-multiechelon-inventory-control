import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import argparse



def parse_log_files(log_directory,test_id):
    """
    Parses all log files in a directory to extract simulation parameters and results.
    """
    all_results = []
    
    # Regex to capture the profit summary lines
    profit_pattern = re.compile(
        r"Average profit <OPT>: (\d+\.?\d*), average profit <(\w+)>: (\d+\.?\d*)"
    )
    # Regex to capture the parameters dictionary
    params_pattern = re.compile(r"Finished Running Set.*?({.*})")
    test_case_name='testcase_'+str(test_id)+'_set_'
    files = [f for f in os.listdir(log_directory) if f.startswith(test_case_name) and f.endswith(".txt")]
    
    for filename in sorted(files, key=lambda x: int(x.split('_')[-1].split('.')[0])):
        print(filename)
        filepath = os.path.join(log_directory, filename)
        with open(filepath, 'r') as f:
            content = f.read()

        # Extract parameters
        params_match = params_pattern.search(content)
        if not params_match:
            print(f"Warning: Could not find parameters in {filename}")
            continue
            
        try:
            # Safely evaluate the dictionary string
            params_dict = ast.literal_eval(params_match.group(1))
            # Extract the non-zero rho value
            rho_val = max(params_dict['rho'][0] + params_dict['rho'][1])
            # Extract ij values for distribution parameters
            ij_val = params_dict.get('ij', [None, None])
        except (ValueError, SyntaxError):
            print(f"Warning: Could not parse parameters dict in {filename}")
            rho_val = -1
            ij_val = [None, None]

        # Extract profit for each policy
        matches = profit_pattern.findall(content)
        
        file_results = {
            'Set': int(filename.split('_')[-1].split('.')[0]), 
            'rho': rho_val,
            'distribt i=1': ij_val[0],
            'distribt j=1': ij_val[1]
        }
        for opt_profit, policy, policy_profit in matches:
            file_results[f"{policy}_Profit"] = float(policy_profit)
            # Store the specific OPT value for this policy's comparison
            file_results[f"{policy}_OPT"] = float(opt_profit)
            
        all_results.append(file_results)
        
    return pd.DataFrame(all_results)

def main():
    """
    Main function to run the parsing and plotting script.
    """
    parser = argparse.ArgumentParser(description="collect inventory management policies results.")
    parser.add_argument('--ncost', action='store_true', help='Use non linear cost')
    parser.add_argument('--testcase_id', type=int, default=2, help='Test case ID to run (0-4), mulitlocation 0 and 1, two location: 2,3,and 4')
    args = parser.parse_args()
    testcase_id = args.testcase_id
    
    log_directory = "./results" # Assumes logs are in a subdirectory named "results"
    if not os.path.isdir(log_directory):
        print(f"Error: Directory '{log_directory}' not found. Please place your log files there.")
        return

    # --- Step 2: Parse the log files into a pandas DataFrame ---
    results_df = parse_log_files(log_directory,testcase_id)
    if results_df.empty:
        print("No data was parsed. Exiting.")
        return

    # --- Step 3: Create the summary table ---
    # Mapping from log names to desired table/plot names
    if args.ncost :
        policy_map = {
            'ADPC': 'ADP',
            'TIEC': 'TIE',
            'LAC': 'LA',
            'NT': 'NT',
            'RCP': 'RC'
        }
    else:
        policy_map = {
            'ADP': 'ADP',
            'TIE': 'TIE',
            'LA': 'LA',
            'NT': 'NT',
            'RCP': 'RC'
        }

    # Select columns for the summary table, including new distribution columns
    summary_cols = {
        'rho': results_df['rho'],
        'distribt i=1': results_df['distribt i=1'],
        'distribt j=1': results_df['distribt j=1']
    }
    # Add OPT column - we can average the OPT values for a representative number
    summary_cols['DP (OPT)'] = results_df[[f"{p}_OPT" for p in policy_map.keys()]].max(axis=1)

    for log_name, display_name in policy_map.items():
        summary_cols[display_name] = results_df[f"{log_name}_Profit"]
        
    summary_df = pd.DataFrame(summary_cols)
    
    # Display the final summary table, grouped by rho and distribution params
    print("\n--- Summary Table of Average Profits ---\n")
    print(summary_df.groupby(['rho', 'distribt i=1', 'distribt j=1']).mean().round(2))

    # --- Step 4: Prepare data for the box plot (this part is unchanged) ---
    # Calculate the profit difference: OPT - Policy Profit
    diff_data = []
    for log_name, display_name in policy_map.items():
        # Calculate difference using the OPT value from the same line
        differences = results_df[f"{log_name}_OPT"] - results_df[f"{log_name}_Profit"]
        for diff in differences:
            diff_data.append({'Policy': display_name, 'Profit Difference': diff})

    plot_df = pd.DataFrame(diff_data)

    # --- Step 5: Generate and display the box plot (this part is unchanged) ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.boxplot(x='Policy', y='Profit Difference', data=plot_df, ax=ax, 
                showmeans=True, meanprops={"marker":"x", "markerfacecolor":"black", "markeredgecolor":"black"})
    
    ax.set_title('Average Profit Difference from Optimal Solution', fontsize=16)
    ax.set_xlabel('Policies', fontsize=12)
    ax.set_ylabel('Average profit difference from optimal solution', fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    print("\nDisplaying box plot... Close the plot window to exit.")
    plt.show()


if __name__ == "__main__":
    main()
