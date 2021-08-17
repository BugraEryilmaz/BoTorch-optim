# BoTorch-optim
## Getting Started
* Minimum python version 3.7
* Required packages listed below
```
pip install botorch
pip install ax-platform
pip install numpy
```
## What's done
### DATCOM optimizasiton
The optimization loop used to optimize the DATCOM simulation results. This part requires a python package for DATCOM GYM environment and DATCOM simulation program.
```
python botorch_datcom.py
```
The first section in the python file provides an interface to change the number of simulations run for the optimization. They can be changed.

When the optimizations finish, the last lines will be the optimum parameters, its corresponding CL/CD value. An example output can be seen below.
```
Parameters: 

{'CHORD1_1': 0.4,
 'CHORD1_2': 0.018103836201984122,
 'CHORD2_1': 0.2464142702240658,
 'CHORD2_2': 0.25,
 'SSPAN1_2': 0.22418681935068174,
 'SSPAN2_2': 0.3,
 'XLE1': 1.25,
 'XLE2': 3.073188177155113}
Best CL/CD: 3.124
```
### BoTorch-Test
A test function, namely branin function, is used for testing and understanding the botorch package.
```
python botorch-test
```
The optimization problem for the brainin function is global minimization. There are 3 global minimum points with value 0.397887. The points for this value can be seen below:
```
(x1 = -π, x2 = 12.275)
(x1 = π, x2 = 2.275)
(x1 = 9.42478, x2 = 2.475)
```
The expected output of this test code is a coordinate close to either of these 3 points with a value close to the minimum value 0.397887. An example output can be seen below.
```
{'x1': 9.414160290290102, 'x2': 2.621994548824227} 0.4227447026320858
```
