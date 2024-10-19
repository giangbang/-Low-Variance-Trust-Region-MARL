# Low variance Trust Region Optimization for multiagent Reinforcement Learning


## Installation

### Install

```shell
pip install -e .
```


### Install Environments Dependencies

**Install SMAC**

Please follow [the official instructions](https://github.com/oxwhirl/smac) to install SMAC.

**Install MPE**

```shell
pip install pettingzoo==1.22.2
pip install supersuit==3.7.0
```

**Install MuJoCo**

First, follow the instructions on https://github.com/openai/mujoco-py, https://www.roboti.us/, and https://github.com/deepmind/mujoco to download the right version of mujoco you need.

Second, `mkdir ~/.mujoco`.

Third, move the .tar.gz or .zip to `~/.mujoco`, and extract it using `tar -zxvf` or `unzip`.

Fourth, add the following line to the `.bashrc`:

```shell
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/<user>/.mujoco/<folder-name, e.g. mujoco210, mujoco-2.2.1>/bin
```

Fifth, run the following command:

```shell
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
pip install mujoco
pip install gym[mujoco]
sudo apt-get update -y
sudo apt-get install -y patchelf
```

**Install Dependencies of MAMuJoCo**

First follow the instructions above to install MuJoCo. Then run the following commands.

```shell
pip install "mujoco-py>=2.1.2.14"
pip install "Jinja2>=3.0.3"
pip install "glfw>=2.5.1"
pip install "Cython>=0.29.28"
```



### Solve Dependencies

After the installation above, run the following commands to solve dependencies.

```shell
pip install gym==0.21.0
pip install pyglet==1.5.0
pip install importlib-metadata==4.13.0
```



## Usage

### Training on Existing Environments

We provide the configuration in the paper in each environments under `run_configs` folder. Users can reproduce our results by using `python -m examples.train --load_config <CONFIG PATH>` and change `<CONFIG PATH>` to the path of the config file on their machine.
