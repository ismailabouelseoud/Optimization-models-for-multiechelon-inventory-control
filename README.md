````markdown
### Thesis

**Ismail Abouelseoud**  
**Topic:** Optimization Models for Multi‑Echelon Inventory Control

This code implements the algorithm from the paper  
> “Approximate Dynamic Programming for Lateral Transshipment Problems in Multi‑Location Inventory Systems”  
> Joern Meissner and Olga V. Senicheva

It reproduces the paper’s results across five experimental setups: three for two-location systems and two for multi-location systems.

## Usage

```bash
python simulate.py --testcase_id {0,1,2,3,4}
````

### Options

1. `--N <int>`
   Number of simulation runs (default: 1000). Runtime scales linearly with this value.

2. `--testcase_id <0–4>`
   Select one of five test cases:

   * `0`: Multi‑location—varying numbers of locations (L) and transshipment costs
   * `1`: Multi‑location—varying numbers of locations (L) and initial inventory configurations
   * `2`: Two locations with negative‑binomial demand
   * `3`: Two locations with Poisson demand
   * `4`: Two locations with uniform demand

3. `--force`
   Forces all simulations to rerun and overwrite existing results.

## Implemented Policies

All policies live in the `policies/` folder:

1. **`dp.py`**
   Exact dynamic programming for two locations (optimal for L=2).

2. **`dp_gen.py`**
   Generalized DP for any number of locations (should match `dp.py` when L=2).

3. **`fadp.py`**
   Forward Adaptive Dynamic Programming (as in the paper).

4. **`lap.py`**
   Lookahead (LA) policy for multi‑location transshipment.

5. **`lpp.py`**
   Real‑time optimal transshipment policy (upper bound for multi‑location considered optimal for multi‑location).

6. **`ntp.py`**
   No‑Transshipment Policy: holds all inventory in its original location.

7. **`rcp.py`**
   RC Policy: moves one unit at a time based on the current inventory state **x**.

8. **`tiep.py`**
   Lateral TIE Policy: proactive heuristic that redistributes when stock falls below expected demand.

## Other Scripts

* **`set_parameters.py`**
  Generates the five parameter sets (three two‑location, two multi‑location).
* **`simulate.py`**
  Main script to run simulations.

## Output

Results are saved under `results/` as `testcase_<id>_set_<set>.txt`. Each file contains the full parameter set and policy outcomes for that run. For example:

```
Finished Running Set 0 from test id 4 with params:
{'Distribution': 'Uniform', 'L': 2, 'T': 4, 'h': [8, 12], 'c': 1,
 'p': [40, 80], 'mu': [0.5, 0.5], 'rho': [[0, 29], [29, 0]], 'S': [3, 3], 'ij': [1, 1]}
Set 0/18, 1000 iterations. Average profit <OPT>: 95.88,
average profit <DPGen>: 95.88. Ratio: 1.0.
Elapsed time: 0.237 sec.
```

In this example, under uniform distribution (`'Uniform'`), two locations (`L = 2`), and four periods (`T = 4`), we have:

* **Holding costs** `h = [8, 12]`
  Cost per unit of leftover inventory at location 1 and location 2, respectively, per period.

* **Transshipment cost factor** `c = 1`
  Cost per unit shipped per unit distance.

* **Unit profits** `p = [40, 80]`
  Revenue earned for each unit sold at location 1 and location 2, respectively.

* **Mean demands** `μ = [0.5, 0.5]`
  Average demand per period at each location (identical here).

* **Distance matrix**

  ```
  ρ = [[ 0, 29],
       [29,  0]]
  ```

  Distance between the two locations (symmetric).

* **Starting inventories** `S = [3, 3]`
  Initial stock levels at each location.

* **`ij = [1, 1]`**
   We draw uniformly from [0, i] and [0, j]; in this example both intervals are [0, 1]. Under test case 4, i and j each range over [1, 2, 3], yielding 18 total parameter sets (indexed 0 through 17 all two location set has 18 sets). 

With these settings, both the exact dynamic program (OPT) and its generalization (DPGen) produce an **average profit of 95.88** (their ratio is **1.0**, indicating perfect agreement) in **0.237 s** of computation time. This confirms that our generalized DP code replicates the optimal two‐location solution from `dp.py` while remaining extensible to larger, multi‐location problems.


```
```
