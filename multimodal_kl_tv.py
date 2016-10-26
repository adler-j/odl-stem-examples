"""Solver for the STEM problem with KL distance and TV norm.

    min_{x1, ..., xn} ||d - sum_i W xi||_2^2 + sum_i kl(di, W xi) + sum_i TV(xi)

This is solved using the douglas_rachford_pd method in ODL.

W is taken as the ray transform.
"""

import numpy as np
import odl

ndim = 2

if ndim == 2:
    # Discrete reconstruction space: discretized functions on the rectangle
    # [-20, 20]^2 with 300 samples per dimension.
    space = odl.uniform_discr(
        min_pt=[-20]*2, max_pt=[20]*2, shape=[128]*2, dtype='float32')

    # Make a parallel beam geometry with flat detector
    # Angles: uniformly spaced, n = 360, min = 0, max = 2 * pi
    angle_partition = odl.uniform_partition(-np.deg2rad(75), np.deg2rad(75), 31)
    # Detector: uniformly sampled, n = 558, min = -30, max = 30
    detector_partition = odl.uniform_partition(-25, 25, 128)
    geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
elif ndim == 3:
    # Discrete reconstruction space: discretized functions on the rectangle
    # [-20, 20]^2 with 300 samples per dimension.
    space = odl.uniform_discr(
        min_pt=[-20]*3, max_pt=[20]*3, shape=[128]*3, dtype='float32')

    # Make a parallel beam geometry with flat detector
    # Angles: uniformly spaced, n = 360, min = 0, max = 2 * pi
    angle_partition = odl.uniform_partition(-np.deg2rad(75), np.deg2rad(75), 31)
    # Detector: uniformly sampled, n = 558, min = -30, max = 30
    detector_partition = odl.uniform_partition([-25]*2, [25]*2, [128]*2)
    geometry = odl.tomo.Parallel3dAxisGeometry(angle_partition, detector_partition,
                                               axis=[1, 0, 0])

# Ray transform (= forward projection). We use ASTRA CUDA backend.
ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

# Create phantom
ellipses = odl.phantom.shepp_logan_ellipses(ndim, modified=True)[::4]

domain = odl.ProductSpace(space, len(ellipses))
phantom = domain.element([odl.phantom.ellipse_phantom(space, [e])
                          for e in ellipses])
phantom = phantom.ufunc.absolute()
phantom.show('phantom', indices=np.s_[:])

diagop = odl.DiagonalOperator(ray_trafo, domain.size)
redop = odl.ReductionOperator(ray_trafo, domain.size)

# Assemble all operators
data = diagop(phantom)
data_sum = redop(phantom)

# gradient
grad = odl.Gradient(ray_trafo.domain)
grad_n = odl.DiagonalOperator(grad, domain.size)

# Create functionals as needed
f = odl.solvers.IndicatorNonnegativity(domain)

alpha = 0.8
g_kl = [(1 - alpha) * odl.solvers.KullbackLeibler(ray_trafo.range, prior=d)
        for d in data]
g_l2 = alpha * odl.solvers.L2NormSquared(ray_trafo.range).translated(data_sum)
g_l1 = [0.01 * odl.solvers.L1Norm(grad.range) for i in range(domain.size)]
g = [odl.solvers.SeparableSum(*g_kl),
     g_l2,
     odl.solvers.SeparableSum(*g_l1)]

opnorm = odl.power_method_opnorm(ray_trafo, maxiter=2)
gradnorm = odl.power_method_opnorm(grad, maxiter=10)
tau = 0.5
sigma = [1 / (opnorm) ** 2, 1 / (ndim * opnorm) ** 2, 1 / (gradnorm) ** 2]

lin_ops = [diagop, redop, grad_n]

# Solve
callback = (odl.solvers.CallbackShow(display_step=10) &
            odl.solvers.CallbackPrintIteration())

x = domain.one()
odl.solvers.douglas_rachford_pd(x, f, g, lin_ops,
                                tau=tau, sigma=sigma,
                                niter=200, callback=callback)
x.show('result', indices=np.s_[:])
