# RecAD
A Pytorch implementation of RecAD.

## Configuration
- Ubuntu 20.04
- NVIDIA driver 460.73.01 
- CUDA 11.2
- Python 3.9
- PyTorch 1.9.0

## Installation
This code requires the packages listed in requirements.txt.
A virtual environment is recommended to run this code

On macOS and Linux:  
```
python3 -m pip install --user virtualenv
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
deactivate
```
Reference: https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/

## Instructions

Clone the template project, replacing ``my-project`` with the name of the project you are creating:

        git clone https://github.com/hanxiao0607/RecAD.git my-project
        cd my-project

Run and test:

        python3 main.py Linear 1 non_causal
        or
        python3 main.py LV 3 non_causal
        or
        python3 main.py MSDS

## Citation
