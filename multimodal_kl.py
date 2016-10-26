"""Solver for the STEM problem with KL distance.

    min_{x1, ..., xn} ||d - sum_i W xi||_2^2 + sum_i kl(di, W xi)

This is solved using the douglas_rachford_pd method in ODL.

W is taken as the ray transform.
"""

import numpy as np
import odl

ndim = 2

if ndim == 2:
    # Discrete reconstruction space
    space = odl.uniform_discr(
        min_pt=[-20]*2, max_pt=[20]*2, shape=[128]*2, dtype='float32')

    # Make a parallel beam geometry with flat detector
    angle_partition = odl.uniform_partition(-np.deg2rad(75), np.deg2rad(75), 31)
    detector_partition = odl.uniform_partition(-25, 25, 128)
    geometry = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)
elif ndim == 3:
    # Discrete reconstruction space
    space = odl.uniform_discr(
        min_pt=[-20]*3, max_pt=[20]*3, shape=[128]*3, dtype='float32')

    # Make a parallel beam geometry with flat detector
    angle_partition = odl.uniform_partition(-np.deg2rad(75), np.deg2rad(75), 31)
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

# Create functionals as needed
f = odl.solvers.IndicatorNonnegativity(domain)

alpha = 0.8
g_kl = [(1 - alpha) * odl.solvers.KullbackLeibler(ray_trafo.range, prior=d)
        for d in data]
g_l2 = alpha * odl.solvers.L2NormSquared(ray_trafo.range).translated(data_sum)
g = [odl.solvers.SeparableSum(*g_kl),
     g_l2]

opnorm = odl.power_method_opnorm(ray_trafo, maxiter=4)
tau = 1.0
sigma = [1 / (opnorm) ** 2, 1 / (ndim * opnorm) ** 2]

# Solve
callback = (odl.solvers.CallbackShow(display_step=10) &
            odl.solvers.CallbackPrintIteration())

x = domain.one()
odl.solvers.douglas_rachford_pd(x, f, g, L=[diagop, redop],
                                tau=tau, sigma=sigma,
                                niter=200, callback=callback)
x.show('result', indices=np.s_[:])
