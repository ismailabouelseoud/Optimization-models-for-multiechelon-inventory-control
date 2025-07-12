````markdown
# ADP Lateral Shipments

This code implements the algorithm described in the paper  
“Approximate Dynamic Programming for Lateral Transshipment Problems in Multi‑Location Inventory Systems”  
by Joern Meissner and Olga V. Senicheva. It reproduces the paper’s results across five experimental setups: three for two‑location systems and two for multi‑location systems.

## Usage

```bash
python simulate.py --testcase_id {0,1,2,3,4}
````

### Options

1. `--N <int>`
   Number of simulation runs (default: 1000). Run time scales with this value.

2. `--testcase_id <0–4>`
   Select one of five test cases:

   * `0`: Multi‑location, varying number of locations (L) and transshipment costs
   * `1`: Multi‑location, varying number of locations (L) and initial inventory configurations
   * `2`: Two locations, negative‑binomial demand
   * `3`: Two locations, Poisson demand
   * `4`: Two locations, uniform demand

3. `--force`
   Forces all simulations to re‑run and overwrite existing results.

## Implemented Policies

1. **`dp.py`**
   Exact dynamic programming for two locations.

2. **`dp_gen.py`**
   Generalized dynamic programming for any number of locations.

3. **`fadp.py`**
   Forward adaptive dynamic programming, as presented in the paper.

4. **`lap.py`**
   Lookahead (LA) policy for multi‑location transshipment.

5. **`lpp.py`**
   Real‑time optimal transshipment policy (upper bound for the multi‑location problem).

6. **`ntp.py`**
   No‑Transshipment Policy: keeps all inventory at original locations.

7. **`rcp.py`**
   RC Policy: for a given inventory state **x**, moves one unit at a time.

8. **`tiep.py`**
   Lateral TIE Policy: a proactive heuristic that triggers redistribution when stock falls below expected demand.

```
```
