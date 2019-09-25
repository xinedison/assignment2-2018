from __future__ import absolute_import, print_function

import tvm
import numpy as np
import topi


# Global declarations of environment.

# llvm
tgt_host="llvm"
# llvm, cuda, opencl, metal
# Change it to respective GPU if gpu is enabled Ex: cuda, opencl
tgt="llvm"


def make_elemwise_add(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    C = tvm.compute(A.shape, lambda *i: A(*i) + B(*i))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_mul(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.placeholder(shape, dtype=dtype, name="B")
    C = tvm.compute(A.shape, lambda *i: A(*i) * B(*i))

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_add_by_const(shape, const_k, tgt, tgt_host, func_name,
                               dtype="float32"):
    """TODO: Your code here"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.const(const_k, dtype=dtype)
    C = tvm.compute(A.shape, lambda *i: A(*i)+B)
    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_elemwise_mul_by_const(shape, const_k, tgt, tgt_host, func_name,
                            dtype="float32"):
    """TODO: Your code here"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.const(const_k, dtype=dtype)
    C = tvm.compute(A.shape, lambda *i: A(*i)*B)
    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_relu(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.max, tvm.const(0, A.dtype)"""
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    B = tvm.const(0, dtype=dtype)
    C = tvm.compute(shape, lambda *i: tvm.max(A(*i), B))
    s = tvm.create_schedule(C.op)
    #print(tvm.lower(s, [A, C], simple_mode=True))
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f



def make_relu_gradient(shape, tgt, tgt_host, func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.select"""
    x = tvm.placeholder(shape, dtype=dtype, name="x")
    x_grad = tvm.placeholder(shape, dtype=dtype, name="x_grad")
    zero = tvm.const(0.0, dtype=dtype)

    y = tvm.compute(shape, lambda *i: tvm.expr.Select(x(*i) > 0, x_grad(*i), 0.0))
    s = tvm.create_schedule(y.op)
    f = tvm.build(s, [x, x_grad, y], tgt, target_host=tgt_host, name=func_name)
    return f

def make_matrix_mul(shapeA, transposeA, shapeB, transposeB, tgt, tgt_host,
                    func_name, dtype="float32"):
    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: treat 4 cases of transposeA, transposeB separately"""
    """Hint: for tvm schedule, use split, reorder, vectorize, parallel"""
    """Hint: debug tvm schedule using tvm.lower"""
    A = tvm.placeholder(shapeA, dtype=dtype, name="A")
    B = tvm.placeholder(shapeB, dtype=dtype, name="B")

    if (not transposeA) and (not transposeB):
        shapeC = (shapeA[0], shapeB[1])
        k = tvm.reduce_axis((0, shapeA[1]), 'k')
        C = tvm.compute(shapeC, lambda x,y : tvm.sum(A[x,k]*B[k,y], axis=k), name='C')
    elif (not transposeA) and transposeB:
        shapeC = (shapeA[0], shapeB[0])
        k = tvm.reduce_axis((0, shapeA[1]), 'k')
        C = tvm.compute(shapeC, lambda x,y : tvm.sum(A[x,k] * B[y, k], axis=k), name='C')
    elif transposeA and (not transposeB):
        shapeC = (shapeA[1], shapeB[1])
        k = tvm.reduce_axis((0, shapeA[0]), 'k')
        C = tvm.compute(shapeC, lambda x,y : tvm.sum(A[k, x] * B[k, y], axis=k), name='C')
    else: # transposeA and transposeB
        shapeC = (shapeA[1], shapeB[0])
        k = tvm.reduce_axis((0, shapeA[0]))
        C = tvm.compute(shapeC, lambda x,y : tvm.sum(A[k,x] * B[y, k], axis=k), name='C')

    s = tvm.create_schedule(C.op)

    block_size = 32
    xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], block_size, block_size)

    k, = s[C].op.reduce_axis
    ko, ki = s[C].split(k, factor=4)
    s[C].reorder(xo, yo, ko, xi, ki, yi)

    # parallel
    s[C].parallel(xo)

    f = tvm.build(s, [A, B, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_conv2d(shapeX, shapeF, tgt, tgt_host, func_name, dtype="float32", padding=0, stride=1):
    assert(shapeX[1] == shapeF[1])
    N, C, H, W = shapeX
    out_c, C, filter_H, filter_W = shapeF


    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum"""
    """Hint: go by conv2d definition. Treat stride=1, padding=0 case only."""
    """For a challenge, treat the general case for stride and padding."""

    assert (H + 2 * padding - filter_H) % stride == 0
    assert (W + 2 * padding - filter_W) % stride == 0
    out_H = (H + 2 * padding - filter_H) // stride + 1
    out_W = (W + 2 * padding - filter_W) // stride + 1
    out_shape = (N, out_c, out_H, out_W)

    x = tvm.placeholder(shapeX, dtype=dtype, name='x')
    feat = tvm.placeholder(shapeF, dtype=dtype, name='f')

    c = tvm.reduce_axis((0, C), name='c')
    k_row = tvm.reduce_axis((0, filter_H), name='filter_h')
    k_col = tvm.reduce_axis((0, filter_W), name='filter_w')

    y = tvm.compute(out_shape, lambda n,m,h,w: tvm.sum(x[n, c, h+k_row, w+k_col] * feat[m, c, k_row, k_col], axis=[k_row, k_col, c]), name='Y')
    s = tvm.create_schedule(y.op)
    #print(tvm.lower(s, [x, feat, y], simple_mode=True))
    f = tvm.build(s, [x, feat, y], tgt, target_host=tgt_host, name=func_name)
    return f


def make_matrix_softmax(shape, tgt, tgt_host, func_name, dtype="float32"):

    """TODO: Your code here"""
    """Hint: use tvm.reduce_axis, tvm.sum, tvm.max, tvm.exp"""
    """Hint: do not reuse the same reduction axis j."""
    """Hint: implement the following version for better stability
        e_x = np.exp(x - np.max(x))
        softmax(x)= e_x / e_x.sum()
    """
    assert len(shape) == 2
    n, cls_dim = shape
    x = tvm.placeholder(shape, dtype=dtype, name='x')
    max_iter = tvm.reduce_axis((0, cls_dim), name='max_iter')
    t_max = tvm.compute((n, ), lambda i: tvm.max(x[i, max_iter], axis=max_iter), name='t_max')
    e_x = tvm.compute(shape, lambda i,j: tvm.exp(x[i,j]-t_max[i]), name='e_x')

    sum_iter = tvm.reduce_axis((0, cls_dim), name='sum_iter')
    e_x_sum = tvm.compute((n,) , lambda i:tvm.sum(e_x[i, sum_iter], axis=sum_iter), name='e_x_sum')
    softmax = tvm.compute(shape, lambda i, j: e_x[i, j] / e_x_sum[i], name='softmax')

    s = tvm.create_schedule(softmax.op)
    f = tvm.build(s, [x, softmax], tgt, target_host=tgt_host, name=func_name)
    return f

    

def make_matrix_softmax_cross_entropy(shape, tgt, tgt_host, func_name,
                                      dtype="float32"):
    """TODO: Your code here"""
    """Hint: output shape should be (1,)"""
    assert len(shape) == 2
    x = tvm.placeholder(shape, dtype=dtype, name='x')
    y = tvm.placeholder(shape, dtype=dtype, name='y')

    n, cls_num = shape
    max_iter = tvm.reduce_axis((0, cls_num), name='max_iter')

    x_max = tvm.compute((n,), lambda i : tvm.max(x[i, max_iter], axis=max_iter))
    e_x = tvm.compute(shape, lambda i,j : tvm.exp(x[i,j] - x_max[i]))

    sum_ex_iter = tvm.reduce_axis((0, cls_num), name='sum_ex_iter')
    e_x_sum = tvm.compute((n,), lambda i : tvm.sum(e_x[i, sum_ex_iter], axis=sum_ex_iter))
    log_softmax = tvm.compute(shape, lambda i,j : tvm.log(e_x[i,j] /e_x_sum[i]))
    y_mul_log_softmax = tvm.compute(shape, lambda i,j: log_softmax[i,j] * y[i,j])

    sum_entropy_iter = tvm.reduce_axis((0, cls_num), name='sum_entropy_iter')
    mean_iter = tvm.reduce_axis((0, n), name='mean_iter')
    sum_entropy = tvm.compute((1,), lambda i:tvm.sum(y_mul_log_softmax[mean_iter, sum_entropy_iter], axis=[mean_iter, sum_entropy_iter]))

    scale = tvm.const(-n, dtype)
    entropy = tvm.compute((1,), lambda i: sum_entropy[i]/scale)


    s = tvm.create_schedule(entropy.op)
    f = tvm.build(s, [x, y, entropy], tgt, target_host=tgt_host, name=func_name)
    return f



def make_reduce_sum_axis_zero(shape, tgt, tgt_host, func_name, dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = topi.sum(A, axis=0, keepdims=False)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_broadcast_to(shape, to_shape, tgt, tgt_host, func_name,
                      dtype="float32"):
    A = tvm.placeholder(shape, dtype=dtype, name="A")
    C = topi.broadcast_to(A, to_shape)

    s = tvm.create_schedule(C.op)
    f = tvm.build(s, [A, C], tgt, target_host=tgt_host, name=func_name)
    return f


def make_sgd_update(shape, learning_rate, tgt, tgt_host, func_name,
                    dtype="float32"):
    X = tvm.placeholder(shape, dtype=dtype, name="A")
    grad = tvm.placeholder(shape, dtype=dtype, name="grad")
    Y = tvm.compute(shape, lambda *i: X(*i) - learning_rate * grad(*i))

    s = tvm.create_schedule(Y.op)
    f = tvm.build(s, [X, grad, Y], tgt, target_host=tgt_host, name=func_name)
    return f
