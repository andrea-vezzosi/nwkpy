# 1 How to install nwkpy 
## 1.1 Download anaconda3

This website provides everything you need to know on how to install anaconda: https://docs.anaconda.com/free/anaconda/install/#

As an alternative, you can install the lighter Miniconda in your LINUX environment.
Following the instructions at https://www.anaconda.com/docs/getting-started/miniconda/install :
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
```
After installing it, close and reopen your terminal application or source the activation script
```
source ~/miniconda3/bin/activate
```
and initialize conda on all available shells with
```
conda init --all
```
Now, with
```
conda list
```
you see the packets installed in the current environment, and with
```
conda env list
```
the available environments, i.e. the environments that you created on your machine.

 
## 1.2 Create a conda environment for nwkpy
Use the following command to create your own environment
```
conda create --name <env-name>
```
(to be consistent among users, <env-name>=nwkp is suggested)

activate the environment
```
conda activate <env-name>
```
and install packages
```
conda install python numpy scipy matplotlib pandas git
```
if you need to perform parallel runs (message passing with MPI), install
```
conda install mpi4py
```
Also, the pip package installer for Python is useful to build the library
```
python3 -m pip install --upgrade build
```

## 1.3 Clone the nwkpy repository on your local machine

### 1.3.1 Create your personal GitHub profile
If you don't have a GitHub personal profile, go to Github.com and create one. Ask the developers to add your profile for access to the nwkpy package.

### 1.3.2 Create your personal token

Create your personal Token 
From your profile, go to the 

[how to clone the repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository)
```
git clone https://github.com/andrea-vezzosi/nwkpy
```
You wil be prompted for your GitHub username. The password is your personale Token
[How to generate your Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens)

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

[how to build a library](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
```
python3 -m pip install --upgrade build
```
To build the library just type
```
cd nwkpy
python -m build
```
Now install the library
```
cd dist
python3 -m pip install nwkpy-0.0.1-py3-none-any.whl
```
check that the library has been installed
```
conda list nwkpy
```
and check that it works
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

# 2 Runnig nwkpy

## 2.1 The mesh file

The main datat of the structure to be simulated is in the mesh file, usually called mesh.msh.
[to be completed]

### 2.1.1 Using FreeFem to generate a mesh

https://freefem.org/
[to be completed]

### 2.1.2 Using nwkpy to generate a mesh

Be carefull that the freefem (lowercase) package that comes installed by default on several Linux distributions os an older version and it is not suitable for nwkpy.
You need to intall FreeFem++. See here for instaructions
https://doc.freefem.org/introduction/installation.html

The script calling FreeFem++ is in

nwkpy/nwkpy/fem/mesh/

## Run a calculation on a serial machine

## Run a calculation on a parallel machine


