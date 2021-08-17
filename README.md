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

When the optimizations finish, the last line will be the optimum parameters, its corresponding reward and CL/CD value (with a small change in environment, so it might print None).
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
The expected output of this test code is a coordinate close to either of these 3 points with a value close to the minimum value 0.397887.
