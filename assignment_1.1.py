import numpy as np
import chaospy as cp
from scipy.integrate import odeint
from matplotlib.pyplot import *
import time

# to perform barycentric interpolation, we'll first compute the barycentric weights
def compute_barycentric_weights(grid):
    size    = len(grid)
    w       = np.ones(size)

    for j in range(1, size):
        for k in range(j):
            diff = grid[k] - grid[j]

            w[k] *= diff
            w[j] *= -diff

    for j in range(size):
        w[j] = 1./w[j]

    return w

# rewrite Lagrange interpolation in the first barycentric form
def barycentric_interp(eval_point, grid, weights, func_eval):
    interp_size = len(func_eval)
    L_G         = 1.
    res         = 0.

    for i in range(interp_size):
        L_G   *= (eval_point - grid[i])

    for i in range(interp_size):
        if abs(eval_point - grid[i]) < 1e-10:
            res = func_eval[i]
            L_G    = 1.0
            break
        else:
            res += (weights[i]*func_eval[i])/(eval_point - grid[i])

    res *= L_G 

    return res

# to use the odeint function, we need to transform the second order differential equation
# into a system of two linear equations
def model(w, t, p):
	x1, x2 		= w
	c, k, f, w 	= p

	f = [x2, f*np.cos(w*t) - k*x1 - c*x2]

	return f

# discretize the oscillator using the odeint function
def discretize_oscillator_odeint(model, atol, rtol, init_cond, args, t, t_interest):
	sol = odeint(model, init_cond, t, args=(args,), atol=atol, rtol=rtol)

	return sol[t_interest, 0]

if __name__ == '__main__':
    # relative and absolute tolerances for the ode int solver
    atol = 1e-10
    rtol = 1e-10

    # parameters setup as specified in the assignement
    c   = 0.5
    k   = 2.0
    f   = 0.5
    y0  = 0.5
    y1  = 0.

    # w is no longer deterministic
    w_left      = 0.95
    w_right     = 1.05
    stat_ref    = [-0.43893703, 0.00019678]

    # create uniform distribution object
    uniform_dist = cp.Uniform(w_left, w_right)

    # no of samples from Monte Carlo sampling
    no_samples_vec = [10, 100, 1000, 10000]
    no_grid_points_vec = [2, 5, 10, 20]

    # time domain setup
    t_max       = 10.
    dt          = 0.01
    grid_size   = int(t_max/dt) + 1
    t           = np.array([i*dt for i in range(grid_size)])
    t_interest  = -1

    # initial conditions setup
    init_cond   = y0, y1

    # create vectors to contain the expectations and variances and runtimes
    expectations    = np.zeros((len(no_grid_points_vec), len(no_samples_vec)))
    variances       = np.zeros((len(no_grid_points_vec), len(no_samples_vec)))
    runtimes        = np.zeros((len(no_grid_points_vec), len(no_samples_vec)))
    mc_expectations = np.zeros((len(no_grid_points_vec), len(no_samples_vec)))
    mc_variances    = np.zeros((len(no_grid_points_vec), len(no_samples_vec)))
    mc_runtimes     = np.zeros((len(no_grid_points_vec), len(no_samples_vec)))

    # compute relative error
    relative_err = lambda approx, real: abs(1. - approx/real)

    # fix seed
    np.random.seed(1234)

    # perform Monte Carlo sampling
for j, no_grid_points in enumerate(no_grid_points_vec):
        # Generate Chebyshev nodes and corresponding function evaluations
        nodes = np.cos((2 * np.arange(1, no_grid_points + 1) - 1) / (2.0 * no_grid_points) * np.pi)
        function_values = np.array([discretize_oscillator_odeint(model, atol, rtol, init_cond, (c, k, f, omega), t, t_interest) for omega in nodes])
        weights = compute_barycentric_weights(nodes)

        for k, no_samples in enumerate(no_samples_vec):
            # a) Create the interpolant and evaluate the integral on the Lagrange interpolant using MC
            start_time = time.time()
            samples = uniform_dist.sample(no_samples)
            interpolated_values = np.array([barycentric_interp(omega, nodes, weights, function_values) for omega in samples])
            expectations[j, k] = np.mean(interpolated_values)
            variances[j, k] = np.var(interpolated_values)
            runtimes[j, k] = time.time() - start_time

            # b) Evaluate the integral directly using MC
            start_time = time.time()
            mc_samples = uniform_dist.sample(no_samples)
            mc_values = np.array([discretize_oscillator_odeint(model, atol, rtol, init_cond, (c, k, f, omega), t, t_interest) for omega in mc_samples])
            mc_expectations[j, k] = np.mean(mc_values)
            mc_variances[j, k] = np.var(mc_values)
            mc_runtimes[j, k] = time.time() - start_time

            # c) Compute expectation and variance and measure runtime
            print(f'Grid Points: {no_grid_points}, Samples: {no_samples}')
            print(f'Interpolant Expectation: {expectations[j, k]}, Variance: {variances[j, k]}, Runtime: {runtimes[j, k]}')
            print(f'MC Expectation: {mc_expectations[j, k]}, Variance: {mc_variances[j, k]}, Runtime: {mc_runtimes[j, k]}')
            print(f'Relative Error in Expectation: {relative_err(expectations[j, k], stat_ref[0])}, MC: {relative_err(mc_expectations[j, k], stat_ref[0])}')
            print(f'Relative Error in Variance: {relative_err(variances[j, k], stat_ref[1])}, MC: {relative_err(mc_variances[j, k], stat_ref[1])}')
            print('-' * 50)