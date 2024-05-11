import copy
import numpy as np
from nwkpy import _constants
from nwkpy._common import Logger
from nwkpy.fem import FemSpace, FemSpaceProduct
from nwkpy.fem.solver import GenEigenProblem
from nwkpy.fem.problem import TotAngMom


class AngularMomentum:
    def __init__(
        self,
        mesh,
        mj_values,
        k=150,
        shape_functions={
            'el': 'Hermite',
            'h' : 'LagrangeQuadratic',
        },
        logger=None
    ):
        self.mesh = mesh
        self.fs_el = FemSpace(copy.copy(mesh), shape_class_name=shape_functions['el'])
        self.fs_h = FemSpace(copy.copy(mesh), shape_class_name=shape_functions['h'])
        self.fsP = FemSpaceProduct(self.fs_el, self.fs_h)

        # number of eigenstates per each mj
        self.k = k

        if logger is not None:
            self.logger = logger
        else:
            self.logger = Logger(rank=0,logfile_path='./')

        self.mj_values = mj_values

    def run(self):

        am = TotAngMom(comps=np.s_[0:8])

        solver = GenEigenProblem()

        solver.assembly(fs=self.fsP, v=am)

        eigvals, eigvecs_el, eigvecs_h, comp_dist, norm_sum_region = solver.solve(
                k=self.k,
                which='LM',
                v0=None,
                sigma=self.mj_values,
                particle='electron',
                eigenvalue_shift=None,
                tol=1e-16
            )
        
        self.eigvals = eigvals
        self.eigvecs_el = eigvecs_el
        self.eigvecs_h = eigvecs_h
        self.spinor_dist = comp_dist
        self.norm_sum_region = norm_sum_region

