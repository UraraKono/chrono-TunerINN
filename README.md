# chrono-TunerINN
Simulator for Tuner INN project using chrono

## Installation of chrono

Clone to the github repo chrono-tunerINN

Try creating an environment from this yaml file in this repo using:
'''bash
conda env create -f environment.yml -n chrono
'''

Download this version of pychrono: https://anaconda.org/projectchrono/pychrono/8.0.0/download/linux-64/pychrono-8.0.0-py39_1.tar.bz2

Once it successfully creates the environment, activate it and do:
'''bash
conda install pychrono-8.0.0-py39_1.tar.bz2
'''

## Code description
* main_map_info.py: Run the simulation on the maps from f1tenth-racetrack that has centerline and raceline

* main.py: Run the simulation on the maps from maps directory

* MPC_VEH_HMMWV.py: It uses steering controller from chrono, which uses position feedback. So it's not real steering feedback.

## chrono_env usage

* Get the state variables: Every time env.step() is called, the state of the vehicle (env.my_hmmwv.state) is updated using 'get_vehicle_state' in chrono_env/utils.py. 


