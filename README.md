This repository contains implementations used for the DDP of Shafeef Omar (ED16B054). 



The code is written in Python 3 and builds on [Tensorflow](https://www.tensorflow.org/). 
Many of the provided reinforcement learning environments require the [Mujoco](http://www.mujoco.org/) physics engine.



### A. Anaconda or Virtualenv

##### A.1. Installing MPI
Ensure that you have a working MPI implementation ([see here](https://mpi4py.readthedocs.io/en/stable/install.html) for more instructions). 

For Ubuntu you can install MPI through the package manager:

```
sudo apt-get install libopenmpi-dev
```

##### A.2. Create either venv or conda environment and activate it

###### Virtualenv
```
pip install --upgrade virtualenv
virtualenv <venv-name>
source <venv-name>/bin/activate
```

###### Anaconda 
If not done yet, install [anaconda](https://www.anaconda.com/) by following the instructions [here](https://www.anaconda.com/download/#linux).
Then reate a anaconda environment, activate it and install the requirements in [`requirements.txt`](requirements.txt).
```
conda create -n <env-name> python=3.6
source activate <env-name>
```

##### B.3. Install the required python dependencies
```
pip install -r requirements.txt
```

##### B.4. Set up the Mujoco physics engine and mujoco-py
For running the majority of the provided Meta-RL environments, the Mujoco physics engine as well as a 
corresponding python wrapper are required.
For setting up [Mujoco](http://www.mujoco.org/) and [mujoco-py](https://github.com/openai/mujoco-py), 
please follow the instructions [here](https://github.com/openai/mujoco-py).



## Running the algorithms

To run the ProMP algorithm in a Mujoco environment with default configurations:
```
python run_scripts/pro-mp_run_mujoco.py 
```


Additionally, in order to run the the gradient-based meta-learning methods MAML ([Finn et. al., 2017](https://arxiv.org/abs/1703.03400)) in a Mujoco environment with the default configuration 
execute, respectively:
```
python run_scripts/maml_run_mujoco.py 
```

## Acknowledgements
This repository includes environments introduced in ([Duan et al., 2016](https://arxiv.org/abs/1611.02779), 
[Finn et al., 2017](https://arxiv.org/abs/1703.03400)), [Rothfuss et al., 2018](https://arxiv.org/abs/1810.06784).
