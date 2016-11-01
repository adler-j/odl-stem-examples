"""Solver for the STEM problem with KL distance and Nuclear norm.

    min_{x1, ..., xn} ||d - sum_i W xi||_2^2 + sum_i kl(di, W xi)
                      + NuclearNorm(grad xi)

This is solved using the douglas_rachford_pd method in ODL.

W is taken as the ray transform.
"""

import numpy as np
import odl

# Select parameters

# Dimension
ndim = 2

# Phantom type
phantom_type = 'circles'

# Select the noise level to use, here very high for the poisson part
# but not very high for the sum of the channels (white noise)
photons_per_pixel = 1  # 1 quanta per pixel, on avg
white_noise_ratio = 0.1  # 10% white noise

# Select how to weight the sum vs the individual channels, higher means
# sum is more important. Can be between 0 and 1
alpha = 0.8

# Regularization method
regularization = 'nuclear'

# Select how strong the regularization should be.
# Should be about 0.1 with the 2d geometry
lam = 0.1

# Select exponent for the norm of the singular vectors if regularization is
# 'nuclear'. Can be set to 1, 2 or 'inf'
exponent = 2


# end of parameters


if ndim == 2:
    # Discrete reconstruction space
    space = odl.uniform_discr(
        min_pt=[-20]*2, max_pt=[20]*2, shape=[128]*2, dtype='float32')

    # Make a parallel beam geometry with flat detector
    angle_partition = odl.uniform_partition(-np.deg2rad(75), np.deg2rad(75),
                                            31)
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
    angle_partition = odl.uniform_partition(-np.deg2rad(75), np.deg2rad(75),
                                            31)
    # Detector: uniformly sampled, n = 558, min = -30, max = 30
    detector_partition = odl.uniform_partition([-25]*2, [25]*2, [128]*2)
    geometry = odl.tomo.Parallel3dAxisGeometry(angle_partition,
                                               detector_partition,
                                               axis=[1, 0, 0])

# Ray transform (= forward projection). We use ASTRA CUDA backend.
ray_trafo = odl.tomo.RayTransform(space, geometry, impl='astra_cuda')

# Create phantom
if phantom_type == 'shepp_logan':
    ellipses = odl.phantom.shepp_logan_ellipses(ndim, modified=True)[::4]

    domain = odl.ProductSpace(space, len(ellipses))
    phantom = domain.element([odl.phantom.ellipse_phantom(space, [e])
                              for e in ellipses])
    phantom = phantom.ufunc.absolute()
elif phantom_type == 'circles':
    ellipses = [[1, 0.8, 0.8, 0, 0, 0],
                [1, 0.4, 0.4, 0.2, 0.2, 0]]

    domain = odl.ProductSpace(space, len(ellipses))
    phantom = domain.element()
    phantom[0] = odl.phantom.ellipse_phantom(space, [ellipses[0]])
    phantom[1] = odl.phantom.ellipse_phantom(space, [ellipses[1]])
    phantom[0] -= phantom[1]

phantom.show('phantom', indices=np.s_[:])

# Create the operators needed
diagop = odl.DiagonalOperator(ray_trafo, domain.size)
redop = odl.ReductionOperator(ray_trafo, domain.size)

# Create data
data = diagop(phantom)
data_sum = redop(phantom)

# Add noise to data.
scale_poisson = photons_per_pixel / np.mean(data)
data = odl.phantom.poisson_noise(data * scale_poisson) / scale_poisson

scale_white_noise = white_noise_ratio * np.mean(data_sum)
data_sum += odl.phantom.white_noise(data_sum.space) * scale_white_noise

# Display data
data.show('data with poisson noise')
data_sum.show('data sum with white noise')

# Create box constraint functional
f = odl.solvers.IndicatorBox(domain, 0, 1)

# Create data discrepancy functionals
g_kl = [(1 - alpha) * odl.solvers.KullbackLeibler(ray_trafo.range, prior=d)
        for d in data]
g_l2 = alpha * odl.solvers.L2NormSquared(ray_trafo.range).translated(data_sum)

# Create regularization functional

# Gradient
grad = odl.Gradient(ray_trafo.domain)
grad_n = odl.DiagonalOperator(grad, domain.size)

if regularization == 'nuclear':
    # Set up the nuclear norm.
    g_reg = lam * odl.solvers.NuclearNorm(grad_n.range,
                                          singular_vector_exp=exponent)
elif regularization == 'tv':
    # Set up l1 norm per dimension
    g_l1 = [lam * odl.solvers.L1Norm(grad.range)] * domain.size

    # Combine by summing
    g_reg = odl.solvers.SeparableSum(*g_l1)

# Assemble functionals
g = [odl.solvers.SeparableSum(*g_kl), g_l2, g_reg]

# Assemble operators
lin_ops = [diagop, redop, grad_n]

# Compute step length parameters to satisfy the condition
# (see douglas_rachford_pd) for more info
tau = 2.0 / len(lin_ops)
sigma = [1 / odl.power_method_opnorm(op, rtol=0.1)**2 for op in lin_ops]

# Create callback for partial results
callback = (odl.solvers.CallbackShow(display_step=10) &
            odl.solvers.CallbackPrintIteration())

# Solve
x = domain.one()
odl.solvers.douglas_rachford_pd(x, f, g, lin_ops,
                                tau=tau, sigma=sigma,
                                niter=200, callback=callback)

# display the final result
x.show('result', indices=np.s_[:])
