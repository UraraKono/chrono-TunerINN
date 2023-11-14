# chrono-TunerINN
Simulator for Tuner INN project using chrono

## Installation of chrono

### Recommended way
Suppose you have CUDA version 12.1 or newer.
1. Clone to the github repo chrono-tunerINN

2. Try creating an environment from this yaml file in this repo using:
```bash
conda env create -f environment.yml -n chrono
```

3. Download this version of pychrono: https://anaconda.org/projectchrono/pychrono/8.0.0/download/linux-64/pychrono-8.0.0-py39_1.tar.bz2

4. Once it successfully creates the environment, activate it and do:
```bash
conda install pychrono-8.0.0-py39_1.tar.bz2
```

### Create empty conda environment first
If there's any additional packages you want to use, you can manually install the packages too.
But this might cause dependencies issues.

1. Create conda environment with python 3.9 (As of November 2023, Numba doesn't support newer python version.)
```bash
conda create --name chrono_env python=3.9
```

2. Install pytorch and pytorch-cuda that corresponds to your CUDA version. (Installing torchvision caused dependencies issues as of Nov 2023 with) following [Pytorch website](https://pytorch.org/get-started/locally/).

3. Install the following packages in this order.
```bash
conda install pyyaml matplotlib gpytorch pyglet
conda install scipy
conda install -c conda-forge mkl=2020
conda install -c conda-forge irrlicht=1.8.5
conda install -c conda-forge pythonocc-core=7.4.1
conda install cuda-toolkit
conda install -c conda-forge glfw
pip install numba
pip install cvxpy
```
4. Install the packages that you want additionally.

5. Download this version of pychrono: https://anaconda.org/projectchrono/pychrono/8.0.0/download/linux-64/pychrono-8.0.0-py39_1.tar.bz2

5. *After you installed all the necessary packages, finally* install Pychrono.
'''bash
conda install pychrono-8.0.0-py39_1.tar.bz2
'''

## Code description
* main_map_info.py: Run the simulation on the maps from f1tenth-racetrack that has centerline and raceline

* main.py: Run the simulation on the maps from maps directory

* MPC_VEH_HMMWV.py: It uses steering controller from chrono, which uses position feedback. So it's not real steering feedback.

## chrono_env usage

* Get the state variables: Every time env.step() is called, the state of the vehicle (env.my_hmmwv.state) is updated using 'get_vehicle_state' in chrono_env/utils.py. 


