import chaospy as cp
import numpy as np

if __name__ == '__main__':
	# define the two distributions
	unif_distr = cp.Uniform(-1, 1)
	norm_distr = cp.Normal(10, 1)

	# degrees of the polynomials
	N = [8]	

	# generate orthogonal polynomials for all N's
	for i, n in enumerate(N):
		
		# employ the three terms recursion scheme using chaospy to generate orthonormal polynomials w.r.t. the two distributions
		# uniform distribution case
		orth_polys_unif = cp.generate_expansion(n, unif_distr, normed=True)
		# normal distribution case
		orth_polys_norm = cp.generate_expansion(n, norm_distr, normed=True)

		# compute <\phi_j(x), \phi_k(x)>_\rho, i.e. E[\phi_j(x) \phi_k(x)]
		# uniform distribution case
		orth_matrix_unif = np.zeros((n+1, n+1))
		for j in range(n+1):
			for k in range(n+1):
				orth_matrix_unif[j, k] = cp.E(orth_polys_unif[j]*orth_polys_unif[k], unif_distr)
		# normal distribution case
		orth_matrix_norm = np.zeros((n+1, n+1))
		for j in range(n+1):
			for k in range(n+1):
				orth_matrix_norm[j, k] = cp.E(orth_polys_norm[j]*orth_polys_norm[k], norm_distr)

		# print result for specific n
		print("For n = %d" % n)
		print("Uniform distribution:")
		np.set_printoptions(precision=3)
		print(orth_matrix_unif)
		print("Normal distribution:")
		np.set_printoptions(precision=3)
		print(orth_matrix_norm)
