import numpy as np
import time
#from __future__ import print_function
from multiprocessing.pool import ThreadPool

def build_dict(keys, values):
    zip_ite = zip(keys, values)
    return dict(zip_ite)

def lorentzian(e, enk, eps):
    t = e-enk
    lor = eps/(eps**2 + t**2) * (1./np.pi)
    return lor

def get_loce(kconec, dlnc_cumul, ncom=8):
    # kconec : conec table of the element
    # kdlnc : number of dof per node (all nodes)
    
    nnel = kconec.shape[0]
    kloce=[]
    for iin in range(nnel):
        ii = kconec[iin]
        ido = dlnc_cumul[ii]*ncom  # % number of cumulative DoF associated with the node 'iin' times components
        idln = dlnc_cumul[ii+1]*ncom - ido
        for iid in range(idln):
            kloce.append(iid+ido)
    return np.array(kloce)

def ortho(A, B):
    """
    Gives a orthonormal matrix, using modified Gram Schmidt Procedure
    :param A: a matrix of column vectors
    :return: a matrix of orthonormal column vectors
    """
    # assuming A is a square matrix
    dim = A.shape[1]
    Q = np.zeros(A.shape, dtype=A.dtype)
    for j in range(0, dim):
        q = A[:,j]
        for i in range(0, j):
            rij = np.vdot(Q[:,i], q)
            rij = Q[:,i].conj().T @ B @ q
            q = q - rij*Q[:,i]
        rjj = q.conj().T @ B @ q
        if np.isclose(rjj,0.0):
            raise ValueError("invalid input matrix")
        else:
            Q[:,j] = q/np.sqrt(rjj)
    return Q


##################### CHECK CONVERGENCE FUNCTION ##############
#def check_convergence(
#    x,
#    y,
#    convergence_criterion='MAE',
#    threshold=1e-3,
#    ):
#
#    assert(x.shape==y.shape, "x and y legths must be equal")
#
#    if convergence_criterion=='MAE':
#        n = x.shape[0]
#        d = np.sum(np.abs(x-y))/n
#    elif convergence_criterion=='MSE':
#        n = x.shape[0]
#        d = np.sum(np.abs(x-y)**2)/n
#    else:
#        # simply compare two numbers

# Einstein summation that uses multithread

"""
Created on Fri Aug 10 09:34:57 2019
@author: Marek Wojciechowski
@github: mrkwjc
@licence: MIT
"""
#from __future__ import print_function
from multiprocessing.pool import ThreadPool

alphabet = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
default_thread_pool = ThreadPool()


def einsumt(*operands, **kwargs):
    """
    Multithreaded version of numpy.einsum function.
    The additional accepted keyword arguments are:
        pool - specifies the pool of processing threads
               If 'pool' is None or it is not given, the default pool with the
               number of threads equal to CPU count is used. If 'pool' is an
               integer, then it is taken as the number of threads and the new
               fresh pool is created. Otherwise, the 'pool' attribute is
               assumed to be a multiprocessing.pool.ThreadPool instance.
        idx  - specifies the subscript along which operands are divided
               into chunks
               Argument 'idx' have to be a single subscript letter, and should
               be contained in the given subscripts, otherwise ValueError is
               risen. If 'idx' is None or it is not given, then the longest
               dimension in operands is searched.
    WARNING: Current implementation allows for string subscripts
             specification only
    """
    pool = kwargs.pop('pool', None)
    idx = kwargs.pop('idx', None)
    # If single processor fall back to np.einsum
    if (pool == 1 or
       (pool is None and default_thread_pool._processes == 1) or
       (hasattr(pool, '_processes') and pool._processes == 1)):
        return np.einsum(*operands, **kwargs)
    # Assign default thread pool if necessary and get number of threads
    if not hasattr(pool, 'apply_async'):
        pool = default_thread_pool  # mp.pool.ThreadPool(pool)
    nproc = pool._processes
    # Out is popped here becaue it is not used in threads but at return only
    out = kwargs.pop('out', None)
    # Analyze subs and ops
    # isubs, osub, ops = np.core.einsumfunc._parse_einsum_input(operands)
    subs = operands[0]
    ops = operands[1:]
    iosubs = subs.split('->')
    isubs = iosubs[0].split(',')
    osub = iosubs[1] if len(iosubs) == 2 else ''
    is_ellipsis = '...' in subs
    if is_ellipsis:
        indices = subs.replace('->', '').replace(',', '').replace('...', '')
        free_indices = ''.join(alphabet - set(indices))
        sis = []
        for si, oi in zip(isubs, ops):
            if '...' in si:
                ne = oi.ndim - len(si.replace('...', ''))  # determine once?
                si = si.replace('...', free_indices[:ne])
            sis.append(si)
        isubs = sis
        osub = osub.replace('...', free_indices[:ne])  # ne is always the same
    if '->' not in subs:  # implicit output
        iss = ''.join(isubs)
        osub = ''.join(sorted([s for s in set(iss) if iss.count(s) == 1]))
    isubs = [s.strip() for s in isubs] # be sure isubs are stripped
    osub = osub.strip() # be sure osub is stripped
    # Get index along which we will chunk operands
    # If not given we try to search for longest dimension
    if idx is not None:  # and idx in indices...
        if idx not in iosubs[0]:
            raise ValueError("Index '%s' is not present in input subscripts"
                             % idx)
        cidx = idx  # given index for chunks
        cdims = []
        for si, oi in zip(isubs, ops):
            k = si.find(cidx)
            cdims.append(oi.shape[k] if k >= 0 else 0)
        if len(set(cdims)) > 2:  # set elements can be 0 and one number
            raise ValueError("Different operand lengths along index '%s'"
                             % idx)
        cdim = max(cdims)  # dimension along cidx
    else:
        maxdim = []
        maxidx = []
        for si, oi in zip(isubs, ops):
            mdim = max(oi.shape)
            midx = si[oi.shape.index(mdim)]
            maxdim.append(mdim)
            maxidx.append(midx)
        cdim = max(maxdim)                    # max dimension of input arrays
        cidx = maxidx[maxdim.index(cdim)]     # index chosen for chunks
    # Position of established index in subscripts
    cpos = [si.find(cidx) for si in isubs]  # positions of cidx in inputs
    opos = osub.find(cidx)                  # position of cidx in output
    ##
    # Determining chunk ranges
    n, r = divmod(cdim, nproc)  # n - chunk size, r - rest
    # Create chunks and apply np.einsum
    n1 = 0
    n2 = 0
    cpos_slice = [(slice(None),)*c for c in cpos]
    njobs = r if n == 0 else nproc
    res = []
    for i in range(njobs):
        args = (subs,)
        n1 = n2
        n2 += n if i >= r else n+1
        islice = slice(n1, n2)
        for j in range(len(ops)):
            oj = ops[j]
            if cpos[j] >= 0:
                jslice = cpos_slice[j] + (islice,)
                oj = oj[jslice]
            args = args + (oj,)
        res += [pool.apply_async(np.einsum, args=args, kwds=kwargs)]
    res = [r.get() for r in res]
    # Reduce
    if opos < 0:  # cidx not in output subs, reducing
        res = np.sum(res, axis=0)
    else:
        res = np.concatenate(res, axis=opos)
    # Handle 'out' and return
    if out is not None:
        out[:] = res
    else:
        out = res
    return res


def bench_einsumt(*operands, **kwargs):
    """
    Benchmark function for einsumt.
    This function returns a tuple 'res' where res[0] is the execution time
    for np.einsum and res[1] is the execution time for einsumt in miliseconds.
    In addition this information is printed to the screen, unless the keyword
    argument pprint=False is set.
    This function accepts all einsumt arguments.
    """
    from time import time
    import platform
    # Prepare kwargs for einsumt
    pprint = kwargs.pop('pprint', True)
    # Preprocess kwargs
    kwargs1 = kwargs.copy()
    pool = kwargs1.pop('pool', None)
    if pool is None:
        nproc = default_thread_pool._processes
        ptype = 'default'
    elif isinstance(pool, int):
        nproc = pool
        ptype = 'custom'
    else:
        nproc = pool._processes
        ptype = 'custom'
    idx = kwargs1.pop('idx', None)
    # np.einsum timing
    t0 = time()
    np.einsum(*operands, **kwargs1)
    dt1 = time() - t0
    N1 = int(divmod(2., dt1)[0])  # we assume 2s of benchmarking
    t0 = time()
    for i in range(N1):
        np.einsum(*operands, **kwargs1)
    dt1 += time() - t0
    T1 = 1000*dt1/(N1+1)
    # einsumt timing
    t0 = time()
    einsumt(*operands, **kwargs)
    dt2 = time() - t0
    N2 = int(divmod(2., dt2)[0])  # we assume 2s of benchmarking
    t0 = time()
    for i in range(N2):
        einsumt(*operands, **kwargs1)
    dt2 += time() - t0
    T2 = 1000*dt2/(N2+1)
    # printing
    if pprint:
        print('Platform:           %s' % platform.system())
        print('CPU type:           %s' % _get_processor_name())
        print('Subscripts:         %s' % operands[0])
        print('Shapes of operands: %s' % str([o.shape
                                              for o in operands[1:]])[1:-1])
        print('Leading index:      %s' % (idx
                                          if idx is not None else 'automatic'))
        print('Pool type:          %s' % ptype)
        print('Number of threads:  %i' % nproc)
        print('Execution time:')
        print('    np.einsum:      %1.4g ms  (average from %i runs)' % (T1,
                                                                        N1+1))
        print('    einsumt:        %1.4g ms  (average from %i runs)' % (T2,
                                                                        N2+1))
        print('Speed up:           %1.3fx' % (T1/T2,))
        print('')
    return T1, T2


def _get_processor_name():
    import os
    import platform
    import subprocess
    import re
    import sys
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        command = "sysctl -n machdep.cpu.brand_string"
        info = subprocess.check_output(command)
        if sys.version_info[0] >= 3:
            info = info.decode()
        return info.strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True)
        if sys.version_info[0] >= 3:
            all_info = all_info.decode()
        for line in all_info.strip().split("\n"):
            if "model name" in line:
                return re.sub(".*model name.*:", "", line, 1).strip()
    return ""





class Logger:
    def __init__(self, rank=0, logfile_path='./'):
        self.rank = rank
        self.logfile_path = logfile_path
        self.logfile_name = logfile_path+"/logfile_ID" + str(rank) + ".txt"

    def logga(self, **kwargs):
        rank = self.rank
        with open(self.logfile_name, "a") as f:
            for key, value in kwargs.items():
                if key == 'size':
                    print("myID = {} \n size = {}".format(rank, value), end='\n\n',file=f)
                elif key == 'kz':
                    print("kz = {}".format(value), end='\n\n',file=f)
                elif key == 'eigenvalues':
                    print("Energies [meV] = {}".format(value),end='\n\n',file=f)
                elif key == 'chemical_potential':
                    print("Chemical Potential [meV] = {}".format(value),end='\n\n',file=f)
                elif key == 'relchargeerror':
                    print("Relative charge error = {}".format(value),end='\n\n',file=f)
                elif key == 'total_charge':
                    print("|Total charge| [cm^-1] = {}".format(value),end='\n\n',file=f)
                elif key == 'max_pot_variation':
                    print("Max potential variation = {}".format(value),end='\n\n',file=f)
                elif key == 'search_energy':
                    print("Search energy (shift-and-invert) = {}".format(value),end='\n\n',file=f)
                elif key == 'iteration_number':
                    print("Current iteration = {}".format(value),end='\n\n',file=f)
                else:
                    print("{} (Unknown keyword) = {}".format(key, value), end='\n\n',file=f)
                    
    def write(self, key='', value='UNKNOWN_VALUE'):
        with open(self.logfile_name, "a") as f:
            print("{}{}".format(key, value), end='\n\n',file=f)


##################### tic() toc() functions #############

def TicTocGenerator():
    # Generator that returns time differences
    ti = 0           # initial time
    tf = time.time() # final time
    while True:
        ti = tf
        tf = time.time()
        yield tf-ti # returns the time difference

TicToc = TicTocGenerator() # create an instance of the TicTocGen generator

# This will be the main function through which we define both tic() and toc()
def toc(tempBool=True):
    # Prints the time difference yielded by generator instance TicToc
    tempTimeInterval = next(TicToc)
    if tempBool:
        print( "Elapsed time: %f seconds.\n" %tempTimeInterval )

def tic():
    # Records a time in TicToc, marks the beginning of a time interval
    toc(False)

