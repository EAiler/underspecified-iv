# underspecified-iv

This repository is the corresponding implementation to Sequential [Underspecified Instrument Selection for Cause-Effect Estimation](https://arxiv.org) (EA, Jason Hartford, Niki Kilbertus).


## Quick Setup
```
git clone git@github.com:eailer/underspecified-iv
cd underspecified-iv
```

The code runs on *Python 3.9*. 
We recommend to create a new environment and to install the packages in the ``requirements.txt`` file.
```
conda create -n underspecified_iv python=3.9
conda install -n underspecified_iv --file spec-file.txt
conda activate underspecified_iv
```


## Reproducing experiments

### Figure 1
Figure 1 shows the finite sample properties of a setting with $d_z=3$, therefore three possible instruments and $d_x=3$, resp. $d_x=10$ potential treatment variables. The causal effect $\beta$ is generated as such that-in principle-can be recovered in full. 

```
python src/run_finite_sample_estimation.py --p 3 --seed 253 --n 1000 --n_runs 500
```
resp.
```
python src/run_finite_sample_estimation.py --p 10 --seed 253 --n 1000 --n_runs 500
```

Each command has three outputs which will be stored in a folder (example for $p=3$)``/Output/p3_n_runs500``:
1. ``p3_n_runs500.npy`` for the scenario i.e. the exact random variables as well as any constants and parameters to recreate the results.
2. ``results.npy``: result file with trajectory values, final optimization value etc.
3. One ``.pdf`` file containing the plot.

### Figure 2 and Figure 3
Figure 2 and Figure 3 show the results of the sequential selection algorithm for the parameter setting $d_z = 30, d_x = 50, d_{id} = 15, N_{IV/exp} = 3$ and $T=6$.

```
python src/run_optimization.py --d 30 --p 50 --d_id 15 --n_runs 500 --n_rounds 6 --d_max 3 --seed 1911
```
This command has three outputs which will be stored in a folder `Output/p50_d30_d_id15_d_max3_n_rounds6`
1. ``p50_d30_d_id15_d_max3_n_rounds6.npy`` i.e. the exact random variables as well as any constants and parameters to recreate the results.
2. ``results.npy``: result file with trajectory values, final optimization value etc.
3. Five ``.pdf`` files containing the plots.

Figures in the **Appendix** are reproduced accordingly.


## Customize

Apart from the reproduction of the experiments, in which we assume ground truth to be known, it is also possible to extract the sequential selection algorithm and customise relevant functionalities to one specific experimental setup example.
The relevant functions are stored in `optimization_submodular.py`. 

There are two classes involved to be addressed.
1. **Experiment**: The function class takes in parameters and outputs a run of the experiment. This has to be exchanged for real data.
2. **SetProposal**: The function class contains the sequential selection. The selection mechanism (``SetProposal()._select()``) can be modifed to integrate different similarity metrics as well as (``SetProposal()._cost_budget()``) to integrate different budget constraints for instrumental variable sets.


