import numpy as np
import os

cdir = os.path.dirname(__file__)
path = cdir+'/outdata/'

############ MESH #############

reg2mat = {
    1 : 'InAs',
    2 : 'GaSb'
}
mat2partic = {
    'InAs'  : 'electron',
    'GaSb' : 'hole'
}

generate_mesh = True
mesh_name = "../ex2/mesh.msh"

R_c = 7.5 # [nm]
shell_width = 4.88 # [nm]
total_width = (R_c+shell_width)*2 # [nm]
print("core radius [nm] = ", R_c) 
print("shell width [nm] = ", shell_width) 
# points per border
edges_per_border = {
    'nC1': 10,
    'nC2': 5,
    'nC3': 7,
    'nC4': 6,
    'nC5': 5,
    'nC6': 5
}
