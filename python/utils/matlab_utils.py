__all__ = [
            "Bundle", "ZEROS_TYPE", "ONES_TYPE", "rad2deg", "deg2rad",
            "realmin", "DEFAULT_ORDER", "mat_like_array", "index_array",
            "quickarray", "ismember", "omin", "omax", "strcmp", "isbundle", 
            "isfield", "cputime","error", "realmax", "eps", "info","warn", 
            "debug", "length","size",  "to_column_mat", "numel", "numDims", 
            "ndims", "expand", "ones", "zeros", "isvector", "isColumnLength",
            "cell",  "iscell", "isnumeric", "isfloat", "isscalar", "is_symmetric", 
            "precc", "succ", "psdpart", "kron", "check_shape",  "sys_integrator",
            "is_pos_def", "sympart", "place_varga", "place", "acker", "is_observable",
            "obsv", "is_controllable", "ctrb", "vec2vecT", "vecv", "specrad", "mdot",
            "smat2", "smat", "mat", "svec2", "Tvec", "svec", "vec"
            ]

__author__ 		= "Lekan Molu"
__copyright__ 	= "2022, Discrete Cosserat SoRO Analysis in Python"
__credits__  	= "There are None."
__license__ 	= "Molux Licence"
__maintainer__ 	= "Lekan Molu"
__email__ 		= "patlekno@icloud.com"
__status__ 		= "Completed"


import time
import math
import numbers
import sys, copy
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla

import logging
# Make sure we have access to the right slycot routines
# try:
#     from slycot import sb03md57
#     # wrap without the deprecation warning
#     def sb03md(n, C, A, U, dico, job='X',fact='N',trana='N',ldwork=None):
#         ret = sb03md57(A, U, C, dico, job, fact, trana, ldwork)
#         return ret[2:]
# except ImportError:
#     try:
#         from slycot import sb03md
#     except ImportError:
#         sb03md = None

# try:
#     from slycot import sb03od
# except ImportError:
#     sb03od = None

logger = logging.getLogger(__name__)

# DEFAULT TYPES
ZEROS_TYPE = np.int64
ONES_TYPE = np.int64
realmin = sys.float_info.min
realmax = sys.float_info.max
eps     = sys.float_info.epsilon
DEFAULT_ORDER = "C"




class Bundle(object):
    def __init__(self, dicko):
        """
            This class creates a Bundle similar to matlab's
            struct class.
        """
        for var, val in dicko.items():
            object.__setattr__(self, var, val)

    def __dtype__(self):
        return Bundle

    def __len__(self):
        return len(self.__dict__.keys())

    def keys():
        return list(self.__dict__.keys())

def mat_like_array(start, end, step=1):
    """
        Generate a matlab-like array start:end
        Subtract 1 from start to account for 0-indexing
    """
    return list(range(start-1, end, step))

def index_array(start=1, end=None, step=1):
    """
        Generate an indexing array for nice slicing of
        numpy-like arrays.
        Subtracts 1 from start to account for 0-indexing
        in python.
    """
    assert end is not None, "end in index array must be an integer"
    return np.arange(start-1, end, step, dtype=np.intp)

def quickarray(start, end, step=1):
    "A quick python array."
    return list(range(start, end, step))


def ismember(a, b):
    "Determines if b is a member of the Hash Table a."
    # See https://stackoverflow.com/questions/15864082/python-equivalent-of-matlabs-ismember-function
    bind = {}
    for i, elt in enumerate(b):
        if elt not in bind:
            bind[elt] = i
    return [bind.get(itm, None) for itm in a]  # None can be replaced by any other "not in b" value

def omin(y, ylast):
    "Determines the minimum among both y and ylast arrays."
    if y.shape == ylast.shape:
        temp = np.vstack((y, ylast))
        return np.min(temp)
    else:
        ylast = expand(ylast.flatten(), 1)
        if y.shape[-1]!=1:
            y = expand(y.flatten(), 1)
        temp = np.vstack((y, ylast))
    return np.min(temp)

def omax(y, ylast):
    "Determines the maximum among both y and ylast arrays."
    if y.shape == ylast.shape:
        temp = np.vstack((y, ylast))
        return np.max(temp)
    else:
        ylast = expand(ylast.flatten(), 1)
        if y.shape[-1]!=1:
            y = expand(y.flatten(), 1)
        temp = np.vstack((y, ylast))
    return np.max(temp)

def strcmp(str1, str2):
    "Compares if strings str1 and atr2 are equal."
    if str1==str2:
        return True
    return False

def isbundle(bund):
    "Determines if bund is an instance of the class Bundle."
    if isinstance(bund, Bundle):
        return True
    return False

def isfield(bund, field):
    "Determines if field is an element of the class Bundle."
    return True if field in bund.__dict__.keys() else False

def cputime():
    "Ad-hoc current time function."
    return time.time()

def error(arg):
    "Pushes std errors out to screen."
    assert isinstance(arg, str), 'logger.fatal argument must be a string'
    raise ValueError(arg)

def info(arg):
    "Pushes std info out to screen."
    assert isinstance(arg, str), 'logger.info argument must be a string'
    logger.info(arg)

def warn(arg):
    "Pushes std watn logs out to screen."
    assert isinstance(arg, str), 'logger.warn argument must be a string'
    logger.warn(arg)

def debug(arg):
    "Pushes std debug logs out to screen."
    assert isinstance(arg, str), 'logger.warn argument must be a string'
    logger.debug(arg)

def length(A):
    "Length of an array A similar to matlab's length function."
    if isinstance(A, list):
        A = np.asarray(A)
    return max(A.shape)

def size(A, dim=None):
    "Size of a matrix A. If dim is specified, returns the size for that dimension."
    if isinstance(A, list):
        A = np.asarray(A)
    if dim is not None:
        return A.shape[dim]
    return A.shape

def to_column_mat(A):
    "Transforms a row-vector array A to a column array."
    n,m = A.shape
    if n<m:
        return A.T
    else:
        return A

def numel(A):
    "Returns the number of elements in an array."
    if isinstance(A, list):
        A = np.asarray(A)
    return np.size(A)

def numDims(A):
    "Returns the numbers of dimensions in an array."
    if isinstance(A, list):
        A = np.asarray(A)
    return A.ndim

def ndims(A):
    "We've got to deprecate this."
    return numDims(A)

def expand(x, ax):
    "Expands an array along axis, ax."
    return np.expand_dims(x, ax)

def ones(rows, cols=None, dtype=ONES_TYPE):
    "Generates a row X col array filled with ones."
    if cols is not None:
        shape = (rows, cols)
    else:
        shape = (rows, rows)
    return np.ones(shape, dtype=dtype)

def zeros(rows, cols=None, dtype=ZEROS_TYPE):
    "Generates a row X col array filled with zeros."
    if cols is not None:
        shape = (rows, cols)
    else:
        if isinstance(rows, tuple):
            shape = rows
        else:
            shape = (rows, rows)
    return np.zeros(shape, dtype=dtype)

def isvector(x):
    "Determines if x is a vector."
    assert numDims(x)>1, 'x must be a 1 x n vector or nX1 vector'
    m,n= x.shape
    if (m==1) or (n==1):
        return True
    else:
        return False

def isColumnLength(x1, x2):
    "Determines if x1 and x2 have the same length along their second dimension."
    if isinstance(x1, list):
        x1 = np.expand_dims(np.asarray(x1), 1)
    return ((ndims(x1) == 2) and (x1.shape[0] == x2) and (x1.shape[1] == 1))

def cell(grid_len, dim=1):
    "Returns a matlab-like cell."
    if dim!=1:
        logger.fatal('This has not been implemented for n>1 cells')
    return [np.nan for i in range(grid_len)]

def iscell(cs):
    "Determines if cs is an instance of a cell."
    if isinstance(cs, list): # or isinstance(cs, np.ndarray):
        return True
    else:
        return False

def isnumeric(A):
    "Determines if A is a numeric type."
    if isinstance(A, numbers.Number):
        return True
    else:
        return False

def isfloat(A):
    "Determines if A is a float type."
    if isinstance(A, np.ndarray):
        dtype = A.dtype
    else:
        dtype = type(A)

    acceptable_types=[np.float64, np.float32, float]

    if dtype in acceptable_types:
        return True
    return False

def isscalar(x):
    "Determines if s is a scalar."
    if (isinstance(x, np.ndarray) and numel(x)==1):
        return True
    elif (isinstance(x, np.ndarray) and numel(x)>1):
        return False
    elif not (isinstance(x, np.ndarray) or isinstance(x, list)):
        return True

def rad2deg(x):
    return (x * 180)/math.pi

def deg2rad(x):
    return x*(math.pi/180)

def vec(A):
    """
        Return the vectorized matrix A by stacking its columns
        on top of one another.

        Input: Matrix A.
        Output: Vectorized form of A.
    """

    return A.reshape(-1, order="F")

def svec(P):
    """
        Return the symmetric vectorization of P i.e.
        the vectorization of the upper triangular part of matrix P.

        Inputs
        ------
        P:  (array) Symmetric matrix in \mathbb{S}^n

        Returns
        -------
        svec(P) = [p_{11} , p_{12} , · · · , p_{1n} , · · · , p_{nn} ]^T

    Author: Lekan Molux, Nov 10, 2022
    """

    assert is_symmetric(P), "P must be a symmetric matrix."

    return P[np.triu_indices(P.shape[0])]

def Tvec(A, shape=None):
    """Returns the vecotization of A.T"""

    assert shape is not None, "shape cannot be None"

    m, n = shape
    res = vec(mat(vec(A), shape=(m, n)).T)
    
    return res

def svec2(P):
    """Return the half-vectorization of matrix P such that its off diagonal entries are doubled.

    Inputs
    ------
    P:  (array) Symmetric matrix in \mathbb{S}^n

    Returns
    -------
    vecs(P):= [p11, 2p12, · · · , 2p1n, p22 , · · · , pnn ]

    Author: Lekan Molux, Nov 10, 2022
    """

    assert is_symmetric(P), "P must be a symmetric matrix."

    T = np.tril(P, 0) + np.triu(P,1)*2

    return T[np.triu_indices(T.shape[0])]

# def svec2(A):
#     """
#         Return the symmetric vectorization i.e. the vectorization
#         of the upper triangular part of matrix A
#         with off-diagonal entries multiplied by sqrt(2)
#         so that la.norm(A, ord='fro')**2 == np.dot(svec2(A), svec2(A))
#     """
    # assert is_symmetric(A), "A must be a symmetric matrix."
#     B = A + np.triu(A)*(2)

#     return B[np.triu_indices(A.shape[0])]


def mat(v, shape=None):
    """Return matricization i.e. the inverse operation of vec of vector v."""
    if shape is None:
        dim = int(np.sqrt(v.size))
        shape = dim, dim
    matrix = v.reshape(shape[1], shape[0]).T
    return matrix    


# def old_mat(v, shape=(m,n)):
#     """
#         Return matricization of vector v i.e. the
#         inverse operation of vec of vector v.

#         This function is deprecated.
#     """
#     assert isinstance(shape, (tuple, list)), "shape must be an instance of list or tuple"
#     m,n = shape
#     matrix = kron(vec(np.eye(n)).T, np.eye(m))@kron(np.eye(n), v)

#     return matrix


def smat(v):
    """
        Return the symmetric matricization of vector v
        i.e. the  inverse operation of svec of vector v.
    """
    m = v.size
    n = int(((1+m*8)**0.5 - 1)/2)
    idx_upper = np.triu_indices(n)
    idx_lower = np.tril_indices(n, -1)
    A = np.zeros([n, n])
    A[idx_upper] = v
    A[idx_lower] = A.T[idx_lower]

    return A    

def smat2(v):
    """
        Return the symmetric matricization of vector v
        i.e. the inverse operation of svec2 of vector v.

        #ToDo: This appears to solve for the case where the 
        off-diag entries are sqrt(V_{mn})
    """
    m = v.size
    n = int(((1+m*8)**0.5 - 1)/2)
    idx_upper = np.triu_indices(n)
    idx_lower = np.tril_indices(n, -1)
    A = np.zeros([n, n])
    A[idx_upper] = v
    A[np.triu_indices(n,1)] /= 2**0.5
    A[idx_lower] = A.T[idx_lower]

    return A

def mdot(*args):
    """Multiple dot product."""

    return reduce(np.dot, args)

def specrad(A):
    """Spectral radius of matrix A."""

    try:
        return np.max(np.abs(la.eig(A)[0]))
    except np.linalg.LinAlgError:
        return np.nan

def vecv(x):
    """Compute the vectorized dot product of x^T and x. Return the """
    xv = kron(x, x)
    ij = np.array(([0]))
    for i in range(1, len(x)+1):
        ij = np.append(ij, np.arange((i-1)*len(x),(i-1)*len(x)+i-1), axis=0)
    ij = ij[1:]
    xv = np.delete(xv, ij, axis=0)

    return xv

def vec2vecT(nr, nc):
    """
        Calculates the transformation matrix from vec(X) to vec(X')

        X has nr row and nc column

        Calling Sig
        -----------
        vec(X') = Tv2v*vec(X)

        Input:
            nr, nc: Numbers of rows and columns respectively.

        Returns:
            T_vt: The transformation matrix.

        Author: Leilei Cui.
    """

    T_vt = np.zeros((nr*nc, nr*nc))

    for i in range(nr):
        for j in range(nc):
            T_vt[i*nc+j,j*nr+i] = 1

    return T_vt


def ctrb(A, B):
    """
        Controllabilty matrix

        Parameters
        ----------
        A, B: array_like
            Dynamics and input matrix of the system

        Returns
        -------
        C: matrix
            Controllability matrix

        Examples
        --------
        >>> C = ctrb(A, B)
    """

    n = np.shape(A)[0]
    C = np.hstack([B] + [la.matrix_power(A, i)@B for i in range(1, n)])

    return C

def is_controllable(A, B):
    """Test if a system is controllable.

        Parameters
        ----------
        A, B: array_like
            Dynamics and input matrix of the system
    """
    ct = ctrb(A, B)

    if la.matrix_rank(ct) != A.shape[0]:
        return False
    else:
        return True


def obsv(C, A):
    """
        Observability matrix

        Parameters
        ----------
        A, C: array_like
            Dynamics and input matrix of the system

        Returns
        -------
        O: matrix
            Controllability matrix

        Examples
        --------
        >>> C = obsv(C, A)
    """

    n = np.shape(A)[0]
    O = np.hstack([C] + [C@la.matrix_power(A, i) for i in range(1, n)])

    return O

def is_observable(A, C):
    """Test if a system is controllable.

        Parameters
        ----------
        A, C: array_like
            State transition and output matrices of the system
    """
    ct = obsv(A, C)

    if la.matrix_rank(ct) != A.shape[0]:
        return False
    else:
        return True

def acker(A, B, poles):
    """
        Pole placement using Ackerman's formula.


        Call:
        K = acker(A, B, poles)

        Parameters
        ----------
        A, B : 2D array_like
            State and input matrix of the system
        poles : 1D array_like
            Desired eigenvalue locations

        Returns
        -------
        K : 2D array (or matrix)
            Gains such that A - B K has given eigenvalues

        Adopted from Richard Murray's code.
    """

    # Make sure the system is controllable
    ct = ctrb(A, B)
    if la.matrix_rank(ct) != A.shape[0]:
        raise ValueError("System not reachable; pole placement invalid")

    # Compute the desired characteristic polynomial
    p = np.real(np.poly(poles))

    # Place the poles using Ackermann's method
    # TODO: compute pmat using Horner's method (O(n) instead of O(n^2))
    n = np.size(p)
    pmat = p[n-1] * la.matrix_power(A, 0)
    for i in np.arange(1, n):
        pmat = pmat + p[n-i-1] * la.matrix_power(A, i)
    K = np.linalg.solve(ct, pmat)

    K = K[-1][:]                # Extract the last row

    return K

# Pole placement
def place(A, B, p):
    """Place closed loop eigenvalues

    K = place(A, B, p)

    Parameters
    ----------
    A : 2D array_like
        Dynamics matrix
    B : 2D array_like
        Input matrix
    p : 1D array_like
        Desired eigenvalue locations

    Returns
    -------
    K : 2D array (or matrix)
        Gain such that A - B K has eigenvalues given in p

    Notes
    -----
    Algorithm
        This is a wrapper function for :func:`scipy.signal.place_poles`, which
        implements the Tits and Yang algorithm [1]_. It will handle SISO,
        MISO, and MIMO systems. If you want more control over the algorithm,
        use :func:`scipy.signal.place_poles` directly.

    Limitations
        The algorithm will not place poles at the same location more
        than rank(B) times.

    The return type for 2D arrays depends on the default class set for
    state space operations.  See :func:`~control.use_numpy_matrix`.

    References
    ----------
    .. [1] A.L. Tits and Y. Yang, "Globally convergent algorithms for robust
       pole assignment by state feedback, IEEE Transactions on Automatic
       Control, Vol. 41, pp. 1432-1452, 1996.

    Examples
    --------
    >>> A = [[-1, -1], [0, 1]]
    >>> B = [[0], [1]]
    >>> K = place(A, B, [-2, -5])

    See Also
    --------
    place_varga, acker

    Notes
    -----
    Lifted from statefdbk function in Murray's python control.
    """
    from scipy.signal import place_poles

    # Convert the system inputs to NumPy arrays
    if (A.shape[0] != A.shape[1]):
        raise ValueError("A must be a square matrix")

    if (A.shape[0] != B.shape[0]):
        err_str = "The number of rows of A must equal the number of rows in B"
        raise ValueError(err_str)

    # Convert desired poles to numpy array
    placed_eigs = np.atleast_1d(np.squeeze(np.asarray(p)))

    result = place_poles(A, B, placed_eigs, method='YT')
    K = result.gain_matrix
    return K


def place_varga(A, B, p, dtime=False, alpha=None):
    """Place closed loop eigenvalues
    K = place_varga(A, B, p, dtime=False, alpha=None)

    Required Parameters
    ----------
    A : 2D array_like
        Dynamics matrix
    B : 2D array_like
        Input matrix
    p : 1D array_like
        Desired eigenvalue locations

    Optional Parameters
    ---------------
    dtime : bool
        False for continuous time pole placement or True for discrete time.
        The default is dtime=False.

    alpha : double scalar
        If `dtime` is false then place_varga will leave the eigenvalues with
        real part less than alpha untouched.  If `dtime` is true then
        place_varga will leave eigenvalues with modulus less than alpha
        untouched.

        By default (alpha=None), place_varga computes alpha such that all
        poles will be placed.

    Returns
    -------
    K : 2D array (or matrix)
        Gain such that A - B K has eigenvalues given in p.

    Algorithm
    ---------
    This function is a wrapper for the slycot function sb01bd, which
    implements the pole placement algorithm of Varga [1]. In contrast to the
    algorithm used by place(), the Varga algorithm can place multiple poles at
    the same location. The placement, however, may not be as robust.

    [1] Varga A. "A Schur method for pole assignment."  IEEE Trans. Automatic
        Control, Vol. AC-26, pp. 517-519, 1981.

    Notes
    -----
    The return type for 2D arrays depends on the default class set for
    state space operations.  See :func:`~control.use_numpy_matrix`.

    Examples
    --------
    >>> A = [[-1, -1], [0, 1]]
    >>> B = [[0], [1]]
    >>> K = place_varga(A, B, [-2, -5])

    See Also:
    --------
    place, acker

    """

    # Make sure that SLICOT is installed
    try:
        from slycot import sb01bd
    except ImportError:
        raise print("can't find slycot module 'sb01bd'")

    # Convert the system inputs to NumPy arrays
    if (A.shape[0] != A.shape[1] or A.shape[0] != B.shape[0]):
        raise ValueError("matrix dimensions are incorrect")

    # Compute the system eigenvalues and convert poles to numpy array
    system_eigs = np.linalg.eig(A)[0]
    placed_eigs = np.atleast_1d(np.squeeze(np.asarray(p)))

    # Need a character parameter for SB01BD
    if dtime:
        DICO = 'D'
    else:
        DICO = 'C'

    if alpha is None:
        # SB01BD ignores eigenvalues with real part less than alpha
        # (if DICO='C') or with modulus less than alpha
        # (if DICO = 'D').
        if dtime:
            # For discrete time, slycot only cares about modulus, so just make
            # alpha the smallest it can be.
            alpha = 0.0
        else:
            # Choosing alpha=min_eig is insufficient and can lead to an
            # error or not having all the eigenvalues placed that we wanted.
            # Evidently, what python thinks are the eigs is not precisely
            # the same as what slicot thinks are the eigs. So we need some
            # numerical breathing room. The following is pretty heuristic,
            # but does the trick
            alpha = -2*abs(min(system_eigs.real))
    elif dtime and alpha < 0.0:
        raise ValueError("Discrete time systems require alpha > 0")

    # Call SLICOT routine to place the eigenvalues
    A_z, w, nfp, nap, nup, F, Z = \
        sb01bd(B.shape[0], B.shape[1], len(placed_eigs), alpha,
               A, B, placed_eigs, DICO)

    # Return the gain matrix, with MATLAB gain convention
    return -F


def sympart(A):
    """
        Return the symmetric part of matrix A.
    """

    return 0.5*(A+A.T)

def is_pos_def(A):
    """Check if matrix A is positive definite."""
    try:
        la.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False

# Utility function to check if a matrix is symmetric
def is_symmetric(M):
    "This from Richard Murray's Python Control Toolbox."
    M = np.atleast_2d(M)
    if isinstance(M[0, 0], np.inexact):
        eps = np.finfo(M.dtype).eps
        return ((M - M.T) < eps).all()
    else:
        return (M == M.T).all()

def succ(A,B):
    """Check the positive definite partial ordering of A > B."""

    return is_pos_def(A-B)

def precc(A,B):
    """Check the negative definite partial ordering of A < B."""

    return not is_pos_def(A-B)

def psdpart(X):
    """Return the positive semidefinite part of a symmetric matrix."""
    X = sympart(X)
    Y = np.zeros_like(X)
    eigvals, eigvecs = la.eig(X)
    for i in range(X.shape[0]):
        if eigvals[i] > 0:
            Y += eigvals[i]*np.outer(eigvecs[:,i],eigvecs[:,i])
    Y = sympart(Y)

    return Y

def kron(*args):
    """Overload and extend the numpy kron function to take a single argument."""
    if len(args)==1:
        return np.kron(args[0], args[0])
    else:
        return np.kron(*args)

# Utility function to check matrix dimensions
def check_shape(name, M, n, m, square=False, symmetric=False):
    "Verify the dims of matrix M."
    if square and M.shape[0] != M.shape[1]:
        raise logger.warn("%s must be a square matrix" % name)

    if symmetric and not is_symmetric(M):
        raise logger.warn("%s must be a symmetric matrix" % name)

    if M.shape[0] != n or M.shape[1] != m:
        raise logger.warn("Incompatible dimensions of %s matrix" % name)

def sys_integrator(sys, X0, K_init, T):
    """Algorithm: to integrate from time 0 to time dt.

        Inputs
        ------
        sys: (Bundle) Object containing A, B, C, D, E matrices.
        X0:  (array) Initial conditions
        K_init:   (array) Initial feedback gain to be used for forced response simulation.
        T : (array), optional for discrete LTI `sys`
            Time steps at which the input is defined; values must be evenly spaced.
    """

    assert isinstance(sys, Bundle), "sys must be of Bundle Type"
    assert isfield(sys, "A"), "Field A is not in sys."
    assert isfield(sys, "B1"), "Field B1 is not in sys."
    assert isfield(sys, "B2"), "Field B2 is not in sys."
    assert isfield(sys, "C"), "Field C is not in sys."
    assert isfield(sys, "D"), "Field D is not in sys."
    assert isfield(sys, "tf"), "Field tf (final time step) is not in sys."
    assert isfield(sys, "dt"), "Field dt (integration time step) is not in sys."

    dt = 1. if sys.dt in [True, None] else sys.dt

    A, B1, B2, C, D = sys.A, sys.B1, sys.B2, sys.C, sys.D

    n_states  = A.shape[0]
    n_inputs  = B1.shape[1]
    n_disturbs  = B2.shape[1]
    n_outputs = C.shape[0]
    n_steps   = T.shape[0]            # number of simulation steps

    input_data = np.zeros((T.shape[0], n_inputs))
    states_data = np.zeros((T.shape[0], n_states))
    states_data[0,:] = X0

    xi   = np.zeros((n_steps, n_inputs)) # exploration input 
    uout = np.zeros((n_steps, n_inputs))

    xout = np.zeros((n_steps, n_states))
    xout[0,:] = X0

    zout = np.zeros((n_steps, n_outputs))

    for i in range(1, n_steps):
        xi[i,:] -= xi[i-1,:]*dt + np.random.normal(0, 1, (1))*np.sqrt(dt) # encourage exploration noise.
        uout[i-1,:] = -K_init@xout[i-1,:] + 10*xi[i,:]
        xout[i,:] = xout[i-1,:] + (A @ xout[i-1,:] + B1 @ uout[i-1,:])*dt + B2 @ xi[i,:]*np.sqrt(dt)
        zout[i,:] = C @ xout[i,:] + D @ uout[i,:]


    return xout, uout, zout