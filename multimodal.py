"""Solver for the STEM problem with L2 distance.

    min_{x1, ..., xn} ||d - sum_i W xi||_2^2 + sum_i ||d - W xi||_2^2

This is solved using the CGLS/CGN method in ODL.

W is taken as the ray transform.
"""

import numpy as np
import odl

# Discrete reconstruction space
space = odl.uniform_discr(
    min_pt=[-20, -20, -20], max_pt=[20, 20, 20], shape=[128]*3, dtype='float32')

# Make a parallel beam geometry with flat detector
angle_partition = odl.uniform_partition(-np.deg2rad(75), np.deg2rad(75), 31)
detector_partition = odl.uniform_partition([-20, -20], [20, 20], [128]*2)
geometry = odl.tomo.Parallel3dAxisGeometry(angle_partition, detector_partition,
                                           axis=[1, 0, 0])

# Ray transform (= forward projection). We use ASTRA CUDA backend.
ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

# Create phantom
ellipses = odl.phantom.shepp_logan_ellipses(3, modified=True)[::4]

domain = odl.ProductSpace(space, len(ellipses))
phantom = domain.element([odl.phantom.ellipse_phantom(space, [e])
                          for e in ellipses])
phantom.show('phantom', indices=np.s_[:])

alpha = 0.8
diagop = (1 - alpha) * odl.DiagonalOperator(ray_trafo, domain.size)
redop = alpha * odl.ReductionOperator(ray_trafo, domain.size)
op = odl.BroadcastOperator(diagop, redop)

data = op(phantom)
data.show('data')

callback = (odl.solvers.CallbackShow('iterates', display_step=5) &
            odl.solvers.CallbackPrintIteration())

x = op.domain.zero()
odl.solvers.conjugate_gradient_normal(op, x, data,
                                      niter=100, callback=callback)
x.show('result', indices=np.s_[:])
