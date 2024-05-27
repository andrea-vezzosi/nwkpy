# 1 How to install nwkpy 
## 1.1 Download anaconda3
This website provides everything you need to know on how to install anaconda: [text](https://docs.anaconda.com/free/anaconda/install/#)
my modification to the readme file

## 1.2 Create your conda environment and install nwkpy
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
python3 -m pip install --upgrade build
```
## 1.3 Clone the repository on your local machine
[how to clone the repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)
```
git clone https://github.com/andrea-vezzosi/nwkpy
```
You wil be prompted for your GitHub username. The password is your personale Token
[How to generate your Token](**https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens**)

## 1.3.1 Create a branch and clone the branch repository on your local machine

You can create a branch from your GitHub page (top-left of your GigHub page to list and create branches) and pull/push that branch, before marging to the main distribution: 

To clone the whole repository (incl ALL branches) and checkout to the specific branch (for pull and push)
```
git clone --branch <branch name> https://github.com/andrea-vezzosi/nwkpy
```

To clone ONLY your branch and checkout to the specific branch (for pull and push)
```
git clone --branch --single-branch <branch name> https://github.com/andrea-vezzosi/nwkpy
```

To clone ONLY the main branch (for pull and push to the main branch)
```
git clone --branch --single-branch main https://github.com/andrea-vezzosi/nwkpy
```

Check which branch is the current target for pull and push
```
git remote show origin
```

## 1.4 build the library
To build the library just type
```
cd nwkpy
python -m build
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
