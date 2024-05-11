# 1 How to install nwkpy 
## 1.1 Download anaconda3
This website provides everything you need to know on how to install anaconda: [text](https://docs.anaconda.com/free/anaconda/install/#)
## 1.2 create your conda environment and install nwkpy
Use the following command to create your own environment
```
conda create --name <env-name>
```
activate the environment
```
conda activate <env-name>
```
and install packages
```
conda install python numpy scipy matplotlib pandas mpi4py git
```
[how to build a library](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
```
% python3 -m pip install --upgrade build
```
To build the library just type
```
% python -m build
```
```
cd dist
python3 -m pip install nwkp-0.0.1-py3-none-any.whl
```
check that the library has been installed
```
conda list nwkpy
```
check that it works
```
python
import nwkpy
```
if you want to uninstall the library
```
pip uninstall nwkpy
```

## make changes and push to the remote server
```
git init -b main
```
type ```ls -a``` to see the .git repo and the .gitignore file \\
to be completed...

## Run a calculation on a serial machine

## Run a calculation on a parallel machine

## Use nwkpy to generate a mesh