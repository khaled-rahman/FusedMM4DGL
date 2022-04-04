"""dgl fusedmm operator module."""
import sys
from itertools import product
from ..backend import gfusedmm as gfusedmm_internal
from .. import backend as F

__all__ = ['gfusedmm', 'fused_cpy_u', 'u_fmul_e_sum']

# this operations are borrowed from spmm's
def reshape_lhs_rhs(lhs_data, rhs_data):
    r""" Expand dims so that there will be no broadcasting issues with different
    number of dimensions. For example, given two shapes (N, 3, 1), (E, 5, 3, 4)
    that are valid broadcastable shapes, change them to (N, 1, 3, 1) and
    (E, 5, 3, 4)
    Parameters
    ----------
    lhs_data : tensor or None
        The left operand, could be None if it's not required by op.
    rhs_data : tensor or None
        The right operand, could be None if it's not required by op.
    """
    lhs_shape = F.shape(lhs_data)
    rhs_shape = F.shape(rhs_data)
    if len(lhs_shape) != len(rhs_shape):
        max_ndims = max(len(lhs_shape), len(rhs_shape))
        lhs_pad_ndims = max_ndims - len(lhs_shape)
        rhs_pad_ndims = max_ndims - len(rhs_shape)
        new_lhs_shape = (lhs_shape[0],) + (1,) * lhs_pad_ndims + lhs_shape[1:]
        new_rhs_shape = (rhs_shape[0],) + (1,) * rhs_pad_ndims + rhs_shape[1:]
        lhs_data = F.reshape(lhs_data, new_lhs_shape)
        rhs_data = F.reshape(rhs_data, new_rhs_shape)
    return lhs_data, rhs_data


def gfusedmm(g, op, reduce_op, lhs_data, rhs_data):
    r""" Generalized FUSEDMM.
    It computes edge features by :attr:`op` lhs features and rhs features.
    
    Parameters
    ----------
    g : DGLGraph
        The input graph.
    op : str
        The binary op's name, could be ``add``, ``sub``, ``mul``, ``div``,
        ``copy_lhs``, ``copy_rhs``.
    reduce_op : str
        Reduce operator, could be ``sum``, ``max``, ``min``, ``mean``.
    lhs_data : tensor or None
        The left operand, could be None if it's not required by the op.
    rhs_data : tensor or None
        The right operand, could be None if it's not required by the op.

    Returns
    -------
    tensor
    
	The result tensor.
    """
    if g._graph.number_of_etypes() == 1:
        if op not in ['fused_cpy_lhs', 'fused_cpy_rhs']:
            lhs_data, rhs_data = reshape_lhs_rhs(lhs_data, rhs_data)
        return gfusedmm_internal(
            g._graph, op, 'sum' if reduce_op == 'mean' else reduce_op, lhs_data, rhs_data)
    else:
        print("Hetero-graph not supported!")
        return None

def _gen_copy_reduce_func(binary_op, reduce_op):

    name = "{}_{}".format(binary_op, reduce_op)
    binary_str = {
        "fused_cpy_u": "It copies node feature to edge as the message.",
        'fused_cpy_e': "It regards edge feature as message."
    }
    x_str = {
        "fused_cpy_u": "source node",
        "fused_cpy_e": "edge"
    }
    def func(g, x):
        if binary_op == 'fused_cpy_u':
            return gfusedmm(g, 'fused_cpy_lhs', reduce_op, x, None)
        elif binary_op == 'u_fmul_e_sum':
            return gfusedmm(g, 'u_fmul_e_sum', reduce_op, x, None)
        else:
            return gfusedmm(g, 'fused_cpy_rhs', reduce_op, None, x)

    func.__name__ = name
    #print("python/dgl/ops/spmm...", name)
    #func.__doc__ = docstring(binary_op)
    return func

def fused_cpy_u(g, x):
    r"""Generalized FusedMM function that copies source node features to edges.

    Parameters
    ----------
    g : DGLHeteroGraph
        The input graph.
    x : tensor
        The source node features.

    Returns
    -------
    tensor
        The result tensor.

    Notes
    -----
    This function supports autograd (computing input gradients given the output gradient).
    """
    return gfusedmm(g, 'fused_cpy_lhs', None, x, None)

def u_fmul_e_sum(g, x):
    r"""Generalized FusedMM function that copies source node features to edges.

    Parameters
    ----------
    g : DGLHeteroGraph
        The input graph.
    x : tensor
        The source node features.

    Returns
    -------
    tensor
        The result tensor.

    Notes
    -----
    This function supports autograd (computing input gradients given the output gradient).
    """
    return gfusedmm(g, 'u_fmul_e_sum', None, x, None)


def _gen_fusedmm_func(binary_op, reduce_op):
    name = "u_{}_e_{}".format(binary_op, reduce_op)
    docstring = r"""Generalized FUSEDMM function.
    It computes edge features by {} features and {} features.
    Parameters
    ----------
    g : DGLHeteroGraph
        The input graph
    x : tensor
        The lhs features.
    y : tensor
        The rhs features.
    Returns
    -------
    tensor
        The result tensor.
    Notes
    -----
    This function supports autograd (computing input gradients given the output gradient). If the
    feature shape of two input operands do not match, we first broadcasts the features to a unified
    shape (note that the memory usage will not increase accordingly) and then performs the operation.
    Broadcasting follows NumPy semantics. Please see
    https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
    for more details about the NumPy broadcasting semantics.
    """.format(binary_op, reduce_op)
    # print("registering fusedmm function:", name)
    def func(g, x, y):
        return gfusedmm(g, binary_op, reduce_op, x, y)
    func.__name__ = name
    # func.__doc__ = docstring
    return func

def _register_fusedmm_func():
    """Register fusedmm functions

    - Binary operation plus reduction between u and e: u_[]_e_[]
    - Copy u plus reduction: copy_u_[]
    - Copy e plus reduction: copy_e_[]
    """
    for binary_op in ["fsub", "fdiv", "fmul", "fadd", "fused_cpy_u", "fused_cpy_e"]:
        for reduce_op in ["sum", "max", "min", "mean"]:
            if binary_op.startswith("fused_cpy"):
                func = _gen_copy_reduce_func(binary_op, reduce_op)
            else:
                func = _gen_fusedmm_func(binary_op, reduce_op)
            setattr(sys.modules[__name__], func.__name__, func)
            __all__.append(func.__name__)

_register_fusedmm_func()
