import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats import qmc
import scipy
from tqdm import tqdm
import pandas as pd
from primitives import weighted_quantile
from primitives import effective_sample_size
import optax
import pickle
import arviz

#### Primitives ####

def softmax1(x):
    """
    Takes a 1D array of size n - 1 and returns a 1D array of size n, taken as its softmax.
    The first coefficient of the softmax is set to 0, hence the difference in size. 
    """
    x_ = jnp.hstack([0, x])
    x_ = jnp.exp(x_)
    return x_ / x_.sum()

def softmax2(x):
    """
    Takes a 2D array of size k x n - 1 and returns a 2D array of size k x n, taken as its
    softmax row-wise. The first coefficient for every row is set to 0, hence the difference
    in size. 
    """

    x_ = jnp.hstack([np.zeros((x.shape[0], 1)), x])
    x_ = jnp.exp(x_)
    return x_ / x_.sum(1)[:,None]

def inv_softmax1(x):
    """
    Takes a 1D array of size n and returns a 1D array of size n - 1, taking the inverse
    softmax transform. 
    """
    C = 0-jnp.log(x[0])
    return (jnp.log(x)+C)[1:]

def inv_softmax2(x):
    """
    Takes a 2D array of size k x n - 1 and returns a 2D array of size k x n, taking the inverse
    softmax transform row-wise. 
    """
    C = 0-jnp.log(x[:,0])
    return (jnp.log(x) + C[:,None])[:,1:]

#### Constructing synthetic dataset ####

def construct_synthetic(true_sampling, I, n):
    """
    Constructs a synthetic dataset for simulation studies on synthetic data, given a system of parameters. 
    
    Inputs:
    - True_sampling: a 1D array containing the probabilities by which the dataset should be sampled.
    It should be provided in 1D/flattened way form, for consistency. 
    - I: shape tuple of the ecological inference application
    - n: a 1D array representing the number of individuals in each unit to simulate. The length of 
    n corresponds to the number of units. 

    """

    K = n.shape[0]
    true_sampling = true_sampling.reshape(I)
    pops = []

    for k in range(K):
        pop = np.random.multinomial(n = n[k], pvals = true_sampling.flatten()).reshape(I)
        pops.append(pop)

    true_data = np.array(pops)
    ecological = jnp.hstack([true_data.sum(2)[:,:-1], true_data.sum(1)[:,:-1]])
    #ecological_full = jnp.hstack([true_data.sum(2), true_data.sum(1)])
    ecological_full = [true_data.sum(2), true_data.sum(1)]
    A, A_full = get_constraint_matrix(I)

    return true_data, ecological, ecological_full, A

def get_constraint_matrix(I):
    """
    Constructs the constraint matrix A for a specific ecological inference application. This enables working
    only in vector representation of X. 

    Takes as input I, the shape tuple of the ecological inference application.
    """
    A1 = jnp.tile(np.diag(np.ones(I[1])),reps=I[0])
    A2 = jnp.repeat(np.diag(np.ones(I[0])),repeats=I[1],axis=1)
    A = jnp.vstack([A2[:-1],A1[:-1]])
    A_full = jnp.vstack([A2,A1])
    return A, A_full

def make_probability_X(size_mat, base=2.):
    """
    Constructs a probability matrix of family type 2. 

    Inputs:
    - size_mat (int): Size of the matrix
    - base (float): asymetry coefficient alpha
    """
    I = (size_mat, size_mat)
    true_sampling = np.zeros(I)
    for i in range(-size_mat+1, size_mat): true_sampling += jnp.diag(jnp.ones(size_mat-jnp.abs(i))*base**(-jnp.abs(i)), k=i)
    true_sampling = (true_sampling/true_sampling.sum()).flatten()
    return true_sampling, I

def make_probability_I(size_mat, base=2.):
    """
    Constructs a probability matrix of family type 2.

    Inputs;
    - size_mat (int): size of the matrix
    - base (float): asymetry coefficient alpha
    """
    I = (size_mat, size_mat)
    true_sampling = np.zeros(I)

    for i in range(size_mat): true_sampling[i,:] += np.ones(size_mat) * (-i) * jnp.log((base))
    #for i in range(size_mat): true_sampling[:,i] += np.ones(size_mat) * jnp.abs(-i) * jnp.log((base/2))

    true_sampling = jnp.exp(true_sampling)
    true_sampling = (true_sampling/true_sampling.sum()).flatten()
    return true_sampling, I

#### Gaussian density approximation ####

def define_prob(pis, n):
    """
    Given the parameters of the multinomial, define its first moment and covariance structure. 

    Inputs:
    - pis (array, 1D): vector of probabilities
    - n (int): size of the multinomial distribution.

    Returns:
    - first moment
    - second moment
    """
    mu0 = pis.copy()
    sigma0 = jnp.array(- pis.reshape(-1,1) @ pis.reshape(1,-1)) + jnp.diag(pis)
    return mu0*n, sigma0*n

def multi_gauss(x, mu, sdet, sinv):
    """
    Computes the log-density of the multivariate normal distribution, with the particularity of 
    taking as input the log-determinant and the inverse of the variance matrix, to speed up 
    computation. 

    Inputs:
    - x (1D array): observation
    - mu (1D array): mean of the Gaussian distribution
    - sdet (float): logarithm of the log-determinant of the covariance matrix of the Gaussian
    distribution
    - sinv (2D array): inverse of the covariance matrix of the Gaussian distribution.

    Returns:
    - log density
    """
    res = -1/2 * (((x-mu).T @ sinv @ (x-mu)) + sdet + sinv.shape[0] * jnp.log(2*jnp.pi))
    return res

def density_gaus(pis, ecological, context, n, A):
    """
    Computes the log-density of the Gaussian approximate model. 
    The output is an array of size K, the number of units, which enables diagnostic of different
    unit contributions to the overall log-density. 

    Inputs:
    - pis (2D array): array of probabilities. Each row corresponds to one observation, and 
    should sum to 1. 
    - ecological (2D array): array of ecological observations. Each row corresponds to one 
    observation. 
    - n (1D array): array of the size of each unit. 
    - A (2D array): constraint matrix

    Outputs:
    - log-density of the Gaussian approximate model (array of size K).
    
    """
    mu, sigma = jax.vmap(define_prob, in_axes=(0, 0))(pis, n)

    Amu = ((A@mu.T).T)
    Asigma = (A@sigma@A.T)
    
    main_term = jax.vmap(lambda ecological, m, sdet, sinv: multi_gauss(ecological, m, sdet, sinv), in_axes=(0, 0, 0, 0))(
        ecological, Amu, jnp.linalg.slogdet(Asigma)[1], jnp.linalg.inv(Asigma))
    
    return main_term

def density_approx(pars, key, pars_func, dens_gaus):
    """
    Computes the log-density of the Gaussian approximate model. The output is of size 1, summing 
    the log-density of all observations. 
    """

    pis = pars_func(pars)
    main_term = dens_gaus(pis)
    return main_term.sum()

#### Marginal likelihood estimation ####

def char_multinom(t, pis, n):
    """
    Log of the characteristic function of a multinomial (n, pis) distribution evaluated in t.

    Inputs:
    - t (complex): point where to evaluate the characteristic function
    - pis (1D array): vector of probabilities.
    - n (int): size of the multinomial distribution

    Output: evaluation of the function.
    """
    return jnp.log((pis * jnp.exp(1j*t)).sum())*n 

def eta_func_(t, pis, obs, n, A):
    """
    Eta (tilde) function evaluated in t. 

    Inputs:
    - t (complex): point where to evaluate the characteristic function
    - pis (1D array): vector of probabilities
    - obs (1D array): vector of one ecological observation
    - n (int): size of the multinomial distribution
    - A: constraint matrix

    Output: evaluation of eta (tilde) function. 
    """
    return jnp.real(jnp.exp(char_multinom(A.T @ t, pis, n)-1j*(t*obs).sum())) * (t.max()<jnp.pi) * (t.min()>-jnp.pi)

eta_func = jax.jit(eta_func_)

def cumulant_multinom_(t, pis, n):
    """
    Cumulant function of a multinomial distribution. 

    Input:
    - t (float): point where to evaluate the cumulant function
    - pis (1D array): vector of probabilities of the multinomial distribution
    - n (int): size of the multinomial distribution

    Output: evaluation of the cumulant distribution.
    """
    return jnp.log((pis * jnp.exp(t)).sum())*n

cumulant_multinom = jax.jit(cumulant_multinom_)

def make_tilt(tilt, pis, A):
    """
    Computes the tilted version of probabilities within the multinomial distribution.

    Inputs:
    - tilt (1D array of size c, the number of constraints): tilting parameters
    - pis (1D array of size d, the dimension of the data X): initial probabilities
    - A constraint matrix (of size c x d)

    Outputs: the tilted version of pis by tilt.
    """
    pis2 = pis * jnp.exp(A.T @ tilt)
    pis2 = pis2/pis2.sum()
    return pis2

def body_func_newton_(tilt, pis, obs, n, A, lr):
    """
    Iteration of Newton to obtain the optimal tilting parameter within the multinomial distribution setting. 

    Inputs:
    - tilt (1D array of size c): initial tilting parameters
    - pis (1D array of size d): initial probabilities for the multinomial distribution
    - obs (1D array of size c): ecological observation
    - n (int): size of the multinomial distribution
    - A: constraint matrix (of size c x d)
    - lr: learning rate

    Returns:
    - The updated tilting parameter. 
    """

    grad_ = (jax.grad(lambda t, p, n, A: cumulant_multinom(A.T @ t, p, n))(tilt, pis, n, A) - obs) #+ (obs < 0.1) * jax.grad(lambda tilt: 1e-3*(tilt**2).sum())(tilt)
    hes_ = jax.hessian(lambda t, p, n, A: cumulant_multinom(A.T @ t, p, n))(tilt, pis, n, A) #+ (obs < 0.1) * jax.hessian(lambda tilt: 1e-3*(tilt**2).sum())(tilt)
    return tilt - lr * jnp.linalg.inv(hes_) @ (grad_)

body_func_newton = jax.jit(body_func_newton_)

def find_optimal_tilt(pis, ecological, n, A, Niter, lr):
    """
    Finds the optimal tilting parameter for a set of observations in the multinomial distribution setting (tau
    function in the paper).

    shapes: K (number of units), c (size of ecological observations), d (size of full data X)

    Inputs:
    - pis (2D array of size K x d): initial probabilities. One row corresponds to one unit and should sum to 1.
    - ecological (2D array of size K x c): ecological observations. One row corresponds to one ecological observation. 
    - n (1D array of size K): size of the multinomial for each unit 
    - A: constraint matrix (of size c x d)
    - Niter: number of Newton iterations
    - lr: learning rate of Newton updates
    """

    tilt = jnp.zeros((ecological.shape[0], A.shape[0]))
    
    body_func = lambda tilt: jax.vmap(body_func_newton, in_axes=(0, 0, 0, 0, None, None))(tilt, pis, ecological, n, A, lr)

    def loop_body(iter_idx, tilt):
        return body_func(tilt)

    final_tilt = jax.lax.fori_loop(0, Niter, loop_body, tilt)
    return final_tilt

def tilted_estimate(key, tilt, pis, obs, n, A, estimate_func):
    """
    Given a (previously optimally set) system of tilting parameters, estimate the marginal likelihood. This only
    considers one ecological observation. 

    Inputs:
    - key: jax random key (size 1)
    - tilt (1D array of size c): tilting parameters
    - pis (1D array of size d): initial probabilities for the multinomial
    - obs (1D array of size c): ecological observation
    - n (int): size of the multinomial
    - A : constraint matrix (of size c x d)
    - estimate_func: function passed used to perform the non-tilted estimation. Corresponds to the initial 
    \hat delta estimators, either uniform or Gaussian.

    Outputs: 
    - log-density for the observation
    """
    pis2 = make_tilt(tilt, pis, A)
    estimate = estimate_func(key, pis2, obs, n, A)
    mult = -(tilt @ obs - cumulant_multinom(A.T @ tilt, pis, n))

    return convert_to_prob(estimate, mult)


def convert_to_prob_(est, mult):
    """
    Given a set of realizations from an estimate function and a set of multiplers (because of the normalization of tilting),
    compute the average and returns the log-density.

    Two technical steps are required to avoid underflowing, either when computing the estimator or when differentiating
    it. Both consist of applying the log-sum-exp trick.

    Inputs:
    - est (array of size K x Nsim): set of simulated values that go inside the non-tilted estimator and 
    which need to be averaged (not in log-space!)
    - mult (array of size K): tilting normalization constants

    Outputs:
    - est_2 (array of size K): the log-density by observations, the main quantity of interest 
    - (jnp.sign(est_a2)>0)*1.: vector coding which weights are negative, used for diagnosis
    - est * jnp.exp(mult): a density component not in log-space (only used for diagnosis)
    - m: second log-sum-exp constant component (only used for diagnosis)
    """
    avoid_underflow = 1e-10
    est_al = jnp.log(jnp.abs(est)+avoid_underflow)
    m = jnp.maximum(est_al.max(), jnp.array([-1e15]))
    est_a2 = (jnp.exp(est_al - m[:,None])).mean() 

    est_2 = jnp.log(est_a2-avoid_underflow) + m + mult - avoid_underflow

    return est_2, (jnp.sign(est_a2)>0)*1., est * jnp.exp(mult), m

convert_to_prob = jax.jit(convert_to_prob_)

def estimate_small(Nsim, key, pis, obs, n, A):
    """
    \hat \delta Uniform estimator (non-tilted version). Estimates without bias (but possibly high variance) the log likelihood.
    This only considers one ecological observation, and outputs observations not in log-space without aggregating them.

    Inputs:
    - Nsim (int): number of simulations
    - key: jax random key of random simulations
    - pis (array of size d): vector of probabilities for the multinomial distribution
    - obs (array of size c): vector of one ecological observation
    - n (int): size of the multinomial distribution
    - A: constraint matrix (of size c x d)

    Outputs:
    - Distinct observations (array of size Nsim), not in log-space. 
    """

    zis = jax.random.uniform(key, shape=(Nsim, A.shape[0],))

    tis = zis * (2*jnp.pi) - jnp.pi
    Xis = jax.vmap(eta_func, in_axes=(0,None,None,None,None))(tis, pis, obs, n, A)

    return Xis

def estimate_medium(Nsim, key, pis, obs, n, A):
    """
    \hat \delta Gaussian estimator (non-tilted version). Estimates without bias (and possibly low variance) the log-likelihood. 
     
    This only considers one ecological observation, and outputs observations not in log-space without aggregating them.

    Inputs:
    - Nsim (int): number of simulations
    - key: jax random key of random simulations
    - pis (array of size d): vector of probabilities for the multinomial distribution
    - obs (array of size c): vector of one ecological observation
    - n (int): size of the multinomial distribution
    - A: constraint matrix (of size c x d)

    Outputs:
    - Distinct observations (array of size Nsim), not in log-space (\eta(Z_i)).
    """

    zis = jax.random.multivariate_normal(key, mean=jnp.zeros(A.shape[0]), cov=jnp.eye(A.shape[0]), shape=(Nsim,))
    
    # Make covariance matrix
    _, sigmainv = define_prob(pis, n)
    sigma = jnp.linalg.inv(A@sigmainv@A.T)

    # Gaussian part 
    tis = zis @ jnp.linalg.cholesky(sigma).T
    wis = jax.scipy.stats.multivariate_normal.pdf(tis, mean=jnp.zeros(sigma.shape[0]), cov=sigma)
    Xis = jax.vmap(eta_func, in_axes=(0,None,None,None,None))(tis, pis, obs, n, A) / wis

    res0 = Xis
    res0 = res0 / (2*jnp.pi)**A.shape[0]

    return res0

def tilted_estimate_qmc(zis, tilt, pis, obs, n, A):
    """
    Performs the same estimation as the tilted_estimate function, except points are not sampled as random,
    but rather using a QMC generator.

    Inputs:
    - key: jax random key (size 1)
    - tilt (1D array of size c): tilting parameters
    - pis (1D array of size d): initial probabilities for the multinomial
    - obs (1D array of size c): ecological observation
    - n (int): size of the multinomial
    - A : constraint matrix (of size c x d)
    - estimate_func: function passed used to perform the non-tilted estimation. Corresponds to the initial 
    \hat delta estimators, either uniform or Gaussian.

    Outputs: 
    - log-density for the observation
    """
    pis2 = make_tilt(tilt, pis, A)
    estimate = estimate_medium_qmc(zis, pis2, obs, n, A)
    mult = -(tilt @ obs - cumulant_multinom(A.T @ tilt, pis, n))

    return convert_to_prob(estimate, mult)

def estimate_medium_qmc(zis, pis, obs, n, A):
    """
    \hat \delta Gaussian estimator (non-tilted version), but instead of randomly drawing points, used provided points earlier,
    possibly using a QMC generator. This only considers one ecological observation, and outputs observations not in log-space without aggregating them.

    Observations cannot be drawn within this function as there is no current implementation of QMC in jax, which
    would make compiling this function using jax.jit impossible.  

    Inputs:
    - zis (array of size Nsim x 1): vector of previously drawn observation
    - pis (array of size d): vector of probabilities for the multinomial distribution
    - obs (array of size c): vector of one ecological observation
    - n (int): size of the multinomial distribution
    - A: constraint matrix (of size c x d)

    Outputs:
    - Distinct observations (array of size Nsim), not in log-space (\eta(Z_i)).
    """
    # Make covariance matrix
    _, sigmainv = define_prob(pis, n)
    sigma = jnp.linalg.inv(A@sigmainv@A.T)

    # Gaussian part 
    tis = zis @ jnp.linalg.cholesky(sigma).T
    wis = jax.scipy.stats.multivariate_normal.pdf(tis, mean=jnp.zeros(sigma.shape[0]), cov=sigma)
    Xis = jax.vmap(eta_func, in_axes=(0,None,None,None,None))(tis, pis, obs, n, A) / wis

    res0 = Xis
    res0 = res0 / (2*jnp.pi)**A.shape[0]

    return res0

def marginal_likelihood(key, pars, pars_func, find_tilt, estimate_prob):
    """
    General wrapper function for computing the tilted version of estimators. This function calls previously defined functions (which can be 
    compileed using jax.jit) to then compute the interest quanity.

    Inputs:
    - key: jax random key for variable generation. 
    - pars: vector of parameters of size p. 
    - pars_func: function which transforms parameters (of size p) into an array of probabilities pis 
    of size K x d. It should be preferably pre-compiled using jit with data arguments passed.  
    - find_tilt: function which transforms pis (of size K x d) into an array of optima tilting parameters
    of size K x c. It should be preferably pre-compiled using jit with data arguments passed.  
    - estimate prob: function which takes a vector of random keys of size K, an array of tilting parameters
    K x c, and an array of probabilities K x d, and computes the log marginal likelihood. It should 
    preferably be pre-compiled using jit (hence the absence of explicit dependency on the data). Its output 
    should be of size K x 1, so diagnosis tools may be used if certain units are problematic. 

    Outputs
    - log marginal likelihood by unit, of size K x 1.
    """
    pis = pars_func(pars)
    tilt = find_tilt(pis)
    keys = jax.random.split(key, pis.shape[0])
    return estimate_prob(keys, tilt, pis)

def marginal_likelihood_qmc(zis, pars, pars_func, find_tilt, estimate_prob_qmc):
    """
    General wrapper function for computing the tilted version of estimators, in QMC version. Instead of the strandard marginal_likelihood function,
    drawn random values should be directly passed to this function.

    Inputs: 
    - zis (array of size K x Nsim): vector of pre-computed random draws (possibly with QMC)
    - pars: vector of parameters of size p. 
    - pars_func: function which transforms parameters (of size p) into an array of probabilities pis 
    of size K x d. It should be preferably pre-compiled using jit with data arguments passed.  
    - find_tilt: function which transforms pis (of size K x d) into an array of optima tilting parameters
    of size K x c. It should be preferably pre-compiled using jit with data arguments passed.  
    - estimate prob_qmc: function which takes an array of random draws of size K x Nsim, an array of tilting parameters
    K x c, and an array of probabilities K x d, and computes the log marginal likelihood. It should 
    preferably be pre-compiled using jit (hence the absence of explicit dependency on the data). Its output 
    should be of size K x 1, so diagnosis tools may be used if certain units are problematic. 

    Outputs
    - log marginal likelihood by unit, of size K x 1.
    """
    pis = pars_func(pars)
    tilt = find_tilt(pis)
    return estimate_prob_qmc(zis, tilt, pis)

def marginal_likelihood_stochastic(key, pars, pars_func, find_tilt, estimate_prob,
                                   ecological_full, ecological, context, n):
    """
    General wrapper function for computing the tilted version of estimators, in stochastic minibatch version. Because of the 
    minibatch, the dependency to the data here is explicit, compared to the standard margina_likelihood function.

    Inputs:
    - key: random rey.
    - pars: vector of parameters of size p. 
    - pars_func: functions that transforms pars, ecological (size K x c), ecological_full (size K x (c + 2)), context (size K x ...),
    n (size K x 1) into an array of probabilities of size K x d. It should preferably be pre-compiled using jit, except that the 
    explicit depencency to the data may not be removed. 
    - find_tilt: function which transforms pis (of size K x d) into an array of optima tilting parameters
    of size K x c. It should be preferably pre-compiled using jit with data arguments passed.  
    - estimate prob: function which takes a vector of random keys of size K, an array of tilting parameters
    K x c, and an array of probabilities K x d, ecological data array (size K x c), and the vector of 
    sizes n (of size K x 1); and computes the log marginal likelihood. It should 
    preferably be pre-compiled using jit (but without removing the depency to the data). Its output 
    should be of size K x 1, so diagnosis tools may be used if certain units are problematic. 

    Outputs:
    - marginal likelihood (size K x 1) by unit.
    """
    
    pis = pars_func(pars, ecological, ecological_full, context, n)
    tilt = find_tilt(pis, ecological, n)
    keys = jax.random.split(key, pis.shape[0])
    return estimate_prob(keys, tilt, pis, ecological, n)

#### Link functions ####

def pars_func_fixed(pars, ecological, ecological_full, context, n, I):
    """
    Parameter function to transform parameters into probabilities
    for the fixed model (model 1).
    
    Inputs:
    - pars (array of size p): vector of parameters
    - ecological (unused)
    - ecological_full: full data array (only uses the number of rows)
    - context (unused)
    - n (unused)
    - I (unused)

    Outputs: probability array pis (of size K x d)
    """
    pis_ = softmax1(pars)
    pis = pis_[None,].repeat(ecological_full.shape[0], 0)
    return pis

def pars_func_margleft(pars, ecological, ecological_full, context, n, I):
    """
    Parameter function to transform parameters into probabilities
    for the model only on conditional probabilities (model 2).

    Inputs:
    - pars (array of size p): vector of parameters
    - ecological (unused)
    - ecological_full: full data array (only uses the number of rows)
    - context (unused)
    - n (unused)
    - I (unused)

    Outputs: probability array pis (of size K x d)
    """
    pbase = (ecological_full[:,:I[0]]/n[:,None]) 
    pbase = pbase / pbase.sum(1)[:,None]
    pcond = softmax2(pars.reshape((I[0], I[1]-1))) 
    pis = pbase[:,:,None]*pcond[None,:,:]
    return pis.reshape((ecological_full.shape[0],-1))

def pars_func_cov1(pars, ecological, ecological_full, context, n, I):
    """
    Parameter function to transform parameters into probabilities
    for the model with conditional probabilities and a covariate (model 3)

    Inputs:
    - pars (array of size p): vector of parameters
    - ecological (unused)
    - ecological_full: full data array (only uses the number of rows)
    - context (array of size K x ...): array of covariates (in the paper, normalized
    log population density)
    - n (unused)
    - I (unused)

    Outputs: probability array pis (of size K x d)
    """
    pbase = (ecological_full[:,:I[0]]/n[:,None])
    pbase = pbase / pbase.sum(1)[:,None]

    pars = pars.reshape((2, I[0], I[1]-1))
    pars2 = pars[0,][None,] + pars[1,][None,] * context.flatten()[:,None,None]
    pcond = jax.vmap(softmax2)(pars2) 

    return (pbase[:,:,None]*pcond).reshape(ecological_full.shape[0], -1)

def define_density(logprior, estimate_function, Nsim, I, ecological_full, ecological, context, n, A, pars_function):
    """
    Help function wrapper to define all the subcomponents used for constructing the estimator of the marginal likelihood. 

    Inputs:
    - logprior: function which takes an array of size p and outputs the logprior density
    - estimate_function: choice for the estimate function (either estimate_medium or estimate_small for the Gaussian or 
    Uniform estimators)
    - Nsim: number of simulations
    - I: size tensor
    - ecological_full (array of size K x (c+2)): full data array
    - ecological (array of size K x c): ecological data 
    - context (array of size K x ...): supplementary covariates data
    - n (array of size n x 1): vector of sizes of the multinomial distribution
    - A: constraint matrix (of size c x d)
    - pars_function: contructor of functions which transforms parameters (of size p) into vector of probabilities. 
    Is for example pars_function_margleft.

    Outputs:
    - dens_exact: estimator of the log-posterior distribution, which takes as input (pars, key). Once constructed, handling
    the estimator is fairly easy.
    - logpost_diag: estimator of the log-marginal likelihood distribution, but without aggregating different units. 
    Used for diagnostic purposes. Takes as input (pars, key)
    """

    pars_func = lambda pars: pars_function(pars, ecological, ecological_full, context, n, I)
    find_tilt = lambda pis: find_optimal_tilt(pis, ecological, n, A, Niter=5, lr=1.)
    est_func = lambda *args: estimate_function(Nsim, *args)
    estimate_prob = lambda keys, tilt, pis: jax.vmap(tilted_estimate, (0, 0, 0, 0, 0, None, None))(keys, tilt, pis, ecological, n, A, est_func)
    dens_exact = lambda pars, key: marginal_likelihood(key, pars, pars_func, find_tilt, estimate_prob)[0].sum() + logprior(pars)
    logpost_diag =  lambda pars, key: marginal_likelihood(key, pars, pars_func, find_tilt, estimate_prob)

    return dens_exact, logpost_diag 

def define_density_stochastic(logprior, estimate_function, Nsim, I, A, pars_function, ratio=1.):
    """
    Help function wrapper to define all the subcomponents used for constructing the estimator of the marginal likelihood;
    for the stochastic minibatch version. Compared to the standard define_density, some explicit dependencies to the
    data are changed. 

    Inputs:
    - logprior: function which takes an array of size p and outputs the logprior density
    - estimate_function: choice for the estimate function (either estimate_medium or estimate_small for the Gaussian or 
    Uniform estimators)
    - Nsim: number of simulations
    - I: size tensor
    - A: constraint matrix (of size c x d)
    - pars_function: contructor of functions which transforms parameters (of size p) into vector of probabilities. 
    Is for example pars_function_margleft.

    Outputs:
    - dens_exact: estimator of the log-posterior distribution, which takes as input (pars, key). Here, there is still
    a dependency to ecological_full (full ecological inference data, size K x (c + 2)), ecological (ecological
    inference data, size K x c), context data (supplementary variables, size K x ...), and size vector n (size K x 1).
    """

    pars_func = lambda pars, ecological, ecological_full, context, n: pars_function(pars, ecological, ecological_full, context, n, I)
    find_tilt = lambda pis, ecological, n: find_optimal_tilt(pis, ecological, n, A, Niter=5, lr=1.)
    est_func = lambda *args: estimate_function(Nsim, *args)
    estimate_prob = lambda keys, tilt, pis, ecological, n: jax.vmap(tilted_estimate, (0, 0, 0, 0, 0, None, None))(keys, tilt, pis, ecological, n, A, est_func)

    dens_exact = lambda pars, key, ecological_full, ecological, context, n: marginal_likelihood_stochastic(key, pars, pars_func, find_tilt, estimate_prob,
                                                                  ecological_full, ecological, context, n)[0].sum() + logprior(pars)*ratio
    
    return dens_exact

def define_density_qmc(logprior, Nsim, I, ecological_full, ecological, context, n, A, pars_function):
    """
    Help function wrapper to define all the subcomponents used for constructing the estimator of the marginal likelihood;
    for the QMC version. Only works for Gaussian estimator (not the uniform one).

    Inputs:
    - logprior: function which takes an array of size p and outputs the logprior density
    - Nsim: number of simulations
    - I: size tensor
    - ecological_full (array of size K x (c+2)): full data array
    - ecological (array of size K x c): ecological data 
    - context (array of size K x ...): supplementary covariates data
    - n (array of size n x 1): vector of sizes of the multinomial distribution
    - A: constraint matrix (of size c x d)
    - pars_function: contructor of functions which transforms parameters (of size p) into vector of probabilities. 
    Is for example pars_function_margleft.

    Outputs: dens_qmc_expensive and dens_qmc_cheap: functions which takes as input pars (array of size P) and 
    a random key and estimates the marginal posterior distribution.

    The difference between the two functions is that:
    - dens_qmc_expensive computes, everytime it is called, a new qmc chain using a Gaussian Sobol generator from
    the scipy.qmc package. As it is not integrated into jax, it is expensive and slow, but it is exact. 
    - dens_qmc_cheap computes a QMC chain only when constructed, and then estimates points by sampling a permutation 
    across different units everytime it is called, which, for large datasets, is practically equivalent to the first
    one, but is much faster. However, it is not exact, from a theoretical point of view.
    """
    
    K = ecological_full.shape[0]
    pars_func = lambda pars: pars_function(pars, ecological, ecological_full, context, n, I)
    find_tilt = lambda pis: find_optimal_tilt(pis, ecological, n, A, Niter=5, lr=1.)
    estimate_prob_qmc = lambda zis, tilt, pis: jax.vmap(tilted_estimate_qmc, (0, 0, 0, 0, 0, None))(zis, tilt, pis, ecological, n, A)
    dens_qmc = jax.jit(lambda pars, zis: marginal_likelihood_qmc(zis, pars, pars_func, find_tilt, estimate_prob_qmc)[0].sum() + logprior(pars))

    def dens_qmc_expensive(pars, key):
        seed = int(jnp.round(jax.random.uniform(key)*1e8))
        zis = jnp.array([qmc.MultivariateNormalQMC(mean=jnp.zeros(A.shape[0]), cov=jnp.eye(A.shape[0])).random(Nsim) for i in range(K)])
        return dens_qmc(pars, zis)

    zis_main = jnp.array([qmc.MultivariateNormalQMC(mean=jnp.zeros(A.shape[0]), cov=jnp.eye(A.shape[0])).random(Nsim) for i in range(K)])

    def dens_qmc_cheap(pars, key):
        zis = jax.random.permutation(key, zis_main, axis=0)
        return dens_qmc(pars, zis)
    
    return jax.jit(dens_qmc_expensive, static_argnums=(1)), dens_qmc_cheap

#### Sampling and Optimization primitives ####

def rw_fixed(x0, key, u0, M, dens_exact):
    """
    Random walk step using a provided estimator of the density function (GIMH).

    Inputs:
    - x0: current value for the parameter
    - key: jax random key
    - u0: current value for the estimated density function.
    - M: mass matrix
    - dens_exact: estimator of the density function. Should take as inputs (pars, key).

    Outputs:
    - New value for the parameter (x1)
    - New value for the estimated density function (u1)
    - accept: whether the propausal was accepted or not.
    """
    keys = jax.random.split(key, 4)

    p0 = jax.random.multivariate_normal(key, mean=jnp.zeros(x0.shape[0]), cov=M, method="svd")
    U = lambda x, key: dens_exact(x, key)

    x1 = x0 + p0
    u1 = U(x1, keys[1])
   
    alpha = u1 - u0 
    accept = jnp.log(jax.random.uniform(keys[2], shape=(1,))) < alpha 
    return x0 + accept*(x1-x0), u0 + accept*(u1-u0), accept

def dens_exact_grad(dens_exact, pars, key):
    """
    Estimator of the gradient of the log-likelihood presented in the article. 

    Inputs:
    - dens_exact: unbiased estimator function of the likelihood, taking as inputs 
    (pars, key)
    - pars: (array of size P)
    - key: random jax key. 

    Outputs:
    - The estimate of the gradient (biased in the scale, but not in the direction)
    """
    val1, grad = jax.value_and_grad(dens_exact)(pars, key)
    val2 = dens_exact(pars, jax.random.fold_in(key, 1))
    return grad * jnp.exp(val1-val2)

def dens_exact_hess(dens_exact, pars, key):
    """
    Estimator of the hessian of the log-likelihood presented in the article, in the same
    spirit of the estimator of the gradient. 

    Inputs:
    - dens_exact: unbiased estimator function of the likelihood, taking as inputs 
    (pars, key)
    - pars: (array of size P)
    - key: random jax key. 

    Outputs:
    - The estimate of the Hessian (biased in the scale, but not in the direction)
    """
    val1 = dens_exact(pars, key)
    hess = jax.hessian(dens_exact)(pars, key)
    val2 = dens_exact(pars, jax.random.fold_in(key, 1))
    return hess * jnp.exp(val1*2-val2*2)

#### Optimization and inference method ####

def find_MAP(dens_function, sizes, solver, dens_function2=None, initial_guess=None, Nmax=1000, Nmax2=0):
    """
    Function to find the MAP using a solver (recommended with Adam).

    Inputs:
    - dens_function: estimator of the log-posterior, taking inputs (pars, key)
    - sizes: sizes object (output of define_functions)
    - solver: optax chosen solver
    - dens_function2 (optional): second estimator of the log-posterior, which can be run for the last 
    iterations if a more costly but more precise estimator is needed to exactly obtain the 
    MAP. Takes inputs (pars, key)
    - initial_guess (optional): initial guess, of size P.
    - Nmax (optional, default 1000): total number of iterations
    - Nmax2 (optional, default 0): number of iterations run with the more precise dens_function2. should 
    be smaller than Nmax.

    Outputs:
    - est_pars: array for the cahin of parameters, of size (Nmax, P)
    - est_vals: array for chain of estimated values of the posterior, of size (Nmax, 1)
    """

    est_pars = np.zeros((Nmax+1, sizes["size_pars"]))

    if initial_guess is not None:
        est_pars[0,]=initial_guess

    est_vals = np.zeros((Nmax+1, 1))
    key = jax.random.key(0)

    compute_loss = lambda pars, key: - dens_function(pars, key)
    
    opt_state = solver.init(est_pars[0,])

    for i in tqdm(range(Nmax)):
        est_vals[i,], grad = jax.value_and_grad(compute_loss)(est_pars[i,], jax.random.fold_in(key,i))
        if jnp.isnan(grad.sum()):
            grad = jnp.zeros_like(grad)
        updates, opt_state = solver.update(grad, opt_state, est_pars[i,])
        est_pars[i+1,] = optax.apply_updates(est_pars[i,], updates)

        if i == Nmax-Nmax2 and Nmax2>0 and dens_function2 is not None:
            compute_loss = lambda pars, key: - dens_function2(pars, key)
    
    est_vals[-1,] = compute_loss(est_pars[-1,], jax.random.fold_in(key,-1))

    return {"pars":est_pars, "vals":est_vals}

def posterior_laplaceIS(mu, dens_func, sizes, type_model, Nmax, Nchains, parallel=False, dens_func_hess=None, mult=1.2):
    """
    Provided a starting point (which needs to be very close to the true MAP), computes the Laplace approximation and performs random weight
    importance sampling to compute the posterior distribution.

    Inputs: 
    - mu (array, size P): starting point 
    - dens_func: estimator of the posterior distribution, takes as input (pars, key).
    - sizes: sizes object (output from define_functions)
    - type_model: type of the model being used (e.g. "fixed", "margleft"). Used when computing aggregates of the posterior distribution.
    - Nmax (int): number of iterations
    - Nchains (int): number of replications of the data. A higher number means faster computation, but a heavier load on the memory of the 
    CPU/GPU. Named Nchains for consistency with the random walk method. The total number of observations will be Nmax x Nchains. 
    - parallel (boolean): if true, the replications are done in parallel on different devices (using jax.pmap). Else, the replications are only
    vectorized (using jax.vmap) on one device.
    - dens_func_hess (optional): possible estimator of the posterior distribution (taking input pars, key). Used for computing the proposal
    IS density Laplace approximation covariance matrix. May be needed for large datasets where dens_func may weigh to much on the momory.
    - mult (float, default 1.2): value by which the Laplace approximation covariance matrix is multiplied.

    Returns a dictionary, composed of
    - "posterior": draws from the Laplace approximation
    - "weights": list [Q, P], composed of Q, the array (of size Nchains x Nmax) of log random weight, and P the array (of size Nchains x Nmax)
    of log propausal weights (of the Laplace approximation)
    - "quants": when possible, the quantiles of the posterior distribution
    - "ESS": importance sampling Random weight. Should be above 200.
    - "Rhat": not used, present for consistency purposes.
    """
    key = jax.random.key(1)

    if dens_func_hess is None:
        dens_func_hess = dens_func

    I = sizes["I"]
    mat = - jax.hessian(dens_func_hess)(mu, jax.random.key(1))/mult
    S = jnp.linalg.inv(mat)

    chain = jax.random.multivariate_normal(key, mean=mu, cov=S, shape=(Nchains*Nmax,), method="svd")

    P = jax.vmap(lambda x: jax.scipy.stats.multivariate_normal.logpdf(x,  mean=mu, cov=S))(chain)

    Qlist = []
    if parallel:
        for i in tqdm(range(Nmax)): 
            Qlist.append(jax.pmap(dens_func)(chain[i*Nchains:(i+1)*Nchains,], jax.random.split(jax.random.key(0), Nchains)).copy()) 
    else:
        for i in tqdm(range(Nmax)): 
            Qlist.append(jax.vmap(dens_func)(chain[i*Nchains:(i+1)*Nchains,], jax.random.split(jax.random.key(0), Nchains)).copy()) 

    Q = jnp.concat(Qlist)

    w = jnp.exp(Q-P - jnp.max(Q-P))
    w = w/w.sum() 

    ESS = (w.sum()**2)/(w**2).sum()
    q = np.array([0.05, 0.5, 0.95])

    if type_model == "margleft":
        chain2 = jax.vmap(softmax2)(chain.reshape((-1, I[0], I[1]-1))).reshape((-1, I[0]*I[1]))
        quants = jax.vmap(weighted_quantile, (1, None, None))(chain2, q, w).reshape((I[0], I[1], 3))
    elif type_model == "fixed":
        chain2 = softmax2(chain).reshape((-1, I[0], I[1]))
        chain2 = chain2/chain2.sum(2)[:,:,None]
        chain2 = chain2.reshape((-1, I[0]*I[1]))
        quants = jax.vmap(weighted_quantile, (1, None, None))(chain2, q, w).reshape((I[0], I[1], 3))
    else: 
        quants=None
    return {"posterior":chain,"weights":[Q,P], "quants":quants, "ESS":ESS, "Rhat":None}

def posterior_RW(mu, dens_func, sizes, type_model, Nmax=300*100, Nchains=1, dens_func_hess=None, parallel=False):
    """
    Provided a starting point, performs pseudo-marginal random walk Monte Caro to obtain the posterior distribution. Uses the Laplace 
    approximation covariance matrix to configurate the transition kernel of the propausals.

    Inputs:
    - mu (array, size P): starting point 
    - dens_func: estimator of the posterior distribution, takes as input (pars, key).
    - sizes: sizes object (output from define_functions)
    - type_model: type of the model being used (e.g. "fixed", "margleft"). Used when computing aggregates of the posterior distribution.
    - Nmax (int): number of iterations
    - Nchains (int): number of independent chains. A higher number means faster computation, but a heavier load on the memory of the 
    CPU/GPU. The total number of observations will be Nmax x Nchains
    - parallel (boolean): if true, the replications are done in parallel on different devices (using jax.pmap). Else, the replications are only
    vectorized (using jax.vmap) on one device.
    - dens_func_hess (optional): possible estimator of the posterior distribution (taking input pars, key). Used for computing transition kernel
    for RW. May be needed for large datasets where dens_func may weigh to much on the momory.
    - mult (float, default .2): value by which the Laplace approximation covariance matrix (used for configuring the 
    transition kernel) is multiplied.

    Outputs a dictionary, with keys corresponding to:
    - "posterior": array of size (Nchains x Nmax) containg the different random walk chains
    - "weights": not used, for consistency purposes only
    - "quants": when possible, the quantiles of the posterior distribution
    - "Rhat": the Rhat statistic computed using the arviz library. Should be very close to 1.
    - "ESS": not used, only present for consistency purposes. 

    """
    key = jax.random.key(1)
    state0 = jnp.ones((Nmax+1, sizes["size_pars"]))
    state0 = state0.at[0,:].set(mu)
    state1 = jnp.zeros((Nmax+1, 1))
    state1 = state1.at[0].set(dens_func(mu, key))
    states = (jnp.repeat(state0[None,:], Nchains, 0), jnp.repeat(state1[None,:], Nchains, 0), jax.random.split(key, Nchains))

    if dens_func_hess is None:
        dens_func_hess = dens_func

    S = jnp.linalg.inv(-jax.hessian(dens_func_hess)(mu, key))*.2

    def loop_body(i, state):
        chain_x0, chain_u0, key = state

        subkey = jax.random.fold_in(key, i)

        x0_next, u0_next, _ = rw_fixed(chain_x0[i,], subkey, chain_u0[i,], S, dens_func)
        chain_x0 = chain_x0.at[i+1,].set(x0_next)
        chain_u0 = chain_u0.at[i+1,].set(u0_next)

        return chain_x0, chain_u0, key
    
    one_chain_fun = jax.jit(lambda state: jax.lax.fori_loop(0, Nmax, loop_body, state))
    
    if parallel:
        states = jax.pmap(one_chain_fun, 0)(states)
    else:
        states = jax.vmap(one_chain_fun, 0)(states)
    chain, _, _ = states

    Rhat = np.array(arviz.rhat(arviz.convert_to_dataset(np.array(chain)))["x"]).max()

    if type_model == "margleft":
        chain2 = np.array(softmax2(chain.reshape((-1,sizes["I"][1]-1))).reshape((Nchains,-1,sizes["I"][0],sizes["I"][1])))
        chain2b = jnp.swapaxes(chain2, 0, 1)
        quants = jnp.quantile(jnp.swapaxes(chain2b.reshape((-1, sizes["I"][0]* sizes["I"][1])), 0,1), q=jnp.array([0.05, 0.5, 0.95]), axis=1)
        quants = quants.reshape((3,  sizes["I"][0], sizes["I"][1]))
        quants = jnp.swapaxes(jnp.swapaxes(quants, 0,2), 0,1)
    else:
        quants = None
        
    return {"posterior":chain,"weights":None,"quants":quants, "Rhat":Rhat, "ESS":None}

#### General study wrapper ####

def define_functions(ecological_data, context_data=None, Nsim=10, estimate_function=estimate_medium, batchsize = 1000, qmc=False, pars_function=pars_func_margleft,
                     A = None):
    """
    Wrapper helper function which constructs the standard, stochastic, and optionally the qmc version of the estimator of the density function. 
    Also constructs the function for the approximate Gaussian model posterior density. This goes from the rough data files to the four functions.

    Takes as input input:
    - ecological_data: a list comprising of dataframes, each with the data for one marginal and with K rows. 
    - context_data: a dataframe with K rows with the context data at the level of units. 
    - Nsim: number of simulations
    - estimate_function: etiher estimate_medium or estimate_small (for Gaussian or uniform sampling)
    - batchsize (int): size of the batch
    - qmc (boolean): if True, generate the cheap qmc version of the estimators 
    - pars_function: function which transforms parameters into probabilities (see above).

    Outputs a dictionary with keys:
    - "dens_exact": standard estimator of the log-posterior function, taking (pars, key) as input
    - "dens_approx": posterior distribution for the approximate Gaussian model. Takes also (pars, key)
    as input, but only for consistency (doesn't use key as it is not random).
    minibatch estimator of the log-posterior function, taking (pars, key) as input
    - "sizes": useful sizes for the analysis. Composed of "size_pars", which corresponds to P, and the 
    shape tuple I (with key "I"). 
    - "dens_stochastic": minibatch estimator of the log-posterior function, taking (pars, key) as input
    - "dens_qmc_cheap": possibly the qmc estimator of the log-posterior function, taking (pars, key) as input. 
    """

    ecological = jnp.hstack([ecological_data[0][:,:-1], ecological_data[1][:,:-1]])
    ecological_full = jnp.hstack([ecological_data[0], ecological_data[1]])

    if context_data is None:
        context = jnp.zeros((ecological.shape[0],1))
    else:
        context = context_data

    n = ecological_data[1].sum(1)
    I = (ecological_data[0].shape[1], ecological_data[1].shape[1])
    K = ecological.shape[0] 
    if A is None:
        A, A_full = get_constraint_matrix(I)

    if pars_function==pars_func_margleft:
        size_pars = (I[0]*(I[1]-1))
    elif pars_function==pars_func_cov1:
        size_pars = (I[0]*(I[1]-1))*2
    else:
        size_pars = (I[0]*(I[1])-1)

    # Handling Parameters 
    mu_prior = jnp.zeros(size_pars)
    s_prior = jnp.eye(size_pars) * 1
    sdet_prior = jnp.linalg.slogdet(s_prior)[1]
    sinv_prior = jnp.linalg.inv(s_prior)

    logprior = jax.jit(lambda pars: multi_gauss(pars, mu_prior, sdet_prior, sinv_prior))
    
    # Constructing functions
    dens_exact_, logpost_diag_ = define_density(logprior, estimate_function, Nsim, I, ecological_full, ecological, context, n, A, pars_function)
    dens_exact = jax.jit(dens_exact_)
    logpost_diag = jax.jit(logpost_diag_)
    
    # Constructing for approximate density
    pars_func = lambda pars: pars_function(pars, ecological, ecological_full, context, n, I)
    dens_gaus = lambda pis: density_gaus(pis, ecological, context, n, A)
    dens_approx = jax.jit(lambda pars, key: density_approx(pars, key, pars_func, dens_gaus) + logprior(pars)) 
    sizes = {"I":I, "size_pars":size_pars}

    # Constructing functions for SGD
    batchset = jnp.arange(0,K)
    batchfun = lambda key: jax.random.choice(key, a=batchset, replace=True, shape=(batchsize,))
    marginal_stochastic = jax.jit(define_density_stochastic(logprior, estimate_function, Nsim, I, A, pars_function, batchsize/K))

    def dens_stochastic(pars, key):
        sel = batchfun(key)
        return marginal_stochastic(pars, key, ecological_full[sel], ecological[sel], context[sel], n[sel])
    
    # Constructing functions with qmc
    dens_qmc_cheap = None
    if qmc:
        dens_qmc_expensive, dens_qmc_cheap = define_density_qmc(logprior, Nsim, I, ecological_full, ecological, context, n, A, pars_function)
    else:
        dens_qmc_cheap = dens_exact

    return {"dens_exact":dens_exact, "dens_approx":dens_approx, "sizes":sizes, 
            "logpost_diag":logpost_diag, "dens_stochastic":dens_stochastic, "dens_qmc_cheap":dens_qmc_cheap}

def make_posterior_pretty(posterior, rows, columns):
    I = posterior.shape
    reshaped_array = jnp.round(posterior,3).reshape(I[0], -1)
    multi_index = pd.MultiIndex.from_product([columns, [".05", ".5", ".95"]], names=['Columns', 'Sub-columns'])
    df = pd.DataFrame(reshaped_array, index=rows, columns=multi_index)
    return df

def full_inference(data, save_as, type_model="margleft", 
                    step1_Nmax = 3000, 
                    step2_Nmax = 5000, step2_Nmax2=1000, step2_lr = 1e-2,
                    step3_Nmax = 1000, step3_Nchains = 5, step3_type = "IS", step3_parallel = False,
                   
                   try_gaussian = False, save_full_posterior=True):
    """
    Full inference loop to conduct the experiments in the real data section of the article. Results
    are saved in pickle format in file save_as.

    Inference is done in three steps:
    1. Optimization on minibatch and then exclusion of problematic units. 
    2. Optimization on full dataset to find the MAP
    3. Computation of the full posterior distribution with either IS or RW.

    Inputs:
    - data: a list composed of dataframes. First two dataframes correspond to the margins of the data, 
    third dataframe (optional) to the additional variables
    - study_name: name of the study for saving the results
    - type_model: type of model (either "fixed", "margleft", or "covariate1")

    - step1_Nmax (int): number of iterations at step 1.
    - step2_Nmax (int): number of iterations at step 2.
    - step2_Nmax (int): number of iterations at step 2 with a more precise estimator of the log-likelihood. 
    - step2_lr (float): learning rate at step2.
    - step3_Nmax (int): number of iterations at step3
    - step3_Nchains (int): number of data duplications at step3. Corresponds for RW as independent chains. 
    - step3_parallel (boolean): if true, data duplications are done on different devices on parallel (using jax.pmap) 
    at step 3. If false, only vectorization happens (using jax.vmap).
    - step3_type (char, either "IS", "RW" or "both"). If "IS", perform random weight importance sampling with 
    the Laplace approximation. If "RW", perform random walk (GIMH). If "both", first perform IS, and if the 
    ESS is below 200, perform RW.

    - try_gaussian (boolean, default False): if true, evaluates the approximate Gaussian model.
    - save_full_posterior (boolean, default True): if False, does not save the full posterior distribution, but
    only quantiles (if available).

    Returns a dictionary with keys corresponding to
    - "step1": output of step1 (see find_MAP)
    - "step2": output of step2 (see find_MAP)
    - "step3": output of step3 (see either posterior_IS or posterior_RW)
    - "est_std": estimation of the standard deviation of the log-likelihood. Should be below 1 for best 
    efficiency.
    """

    result = dict()

    if type_model == "margleft":
        pars_function = pars_func_margleft
    elif type_model == "fixed":
        pars_function = pars_func_fixed
    elif type_model == "covariate1":
        pars_function = pars_func_cov1
    else:
        return None

    if len(data)>2:
        context_data = data[2].to_numpy().astype(float)
    else: 
        context_data = jnp.zeros((data[0].shape[0],1))

    base = .2
    ecological_data = [data[0].to_numpy().astype(float), data[1].to_numpy().astype(float)]
    ecological_data = [jnp.maximum(ecological_data[0], jnp.array([base])), jnp.maximum(ecological_data[1], jnp.array([base]))]

    # Step 1: Put .2 on values zero, fast SGD for warm start and removing weird units
    print("Step 1")
    batchsize = min(2000, data[0].shape[0]//2)
    funcs0 = define_functions(ecological_data, context_data, Nsim=2**4, batchsize=batchsize, qmc=False, pars_function=pars_function)

    if step1_Nmax>0:
        step1 = find_MAP(funcs0["dens_stochastic"], funcs0["sizes"], optax.adam(learning_rate=1e-1), Nmax=step1_Nmax, Nmax2 = 0)
        mu = step1["pars"][-1,]
        crit1 = jnp.isnan(funcs0["logpost_diag"](mu, jax.random.key(3))[0]).flatten()
        
        est_std = [jax.vmap(funcs0["logpost_diag"])(mu[None,].repeat(5, 0), jax.random.split(jax.random.key(i), 5)) for i in range(20)]
        est_std = jnp.array([est_std[i][0] for i in range(20)]).reshape(100,-1).std(0)
        crit2 = est_std > 1

        crit = crit1 | crit2
        problem = jnp.where(crit)[0]
        print(problem)
        print(problem.shape[0])
        no_problem = jnp.where(~crit)[0]
        ecological_data = [ecological_data[0][no_problem,], ecological_data[1][no_problem,]]
        context_data = context_data[no_problem,]
    else:
        mu = jnp.zeros(funcs0["sizes"]["size_pars"])
        step1 = None
    
    
    # Step 2: Running the full inference loop at scale
    
    print("Step 2")
    funcs1 = define_functions(ecological_data, context_data, Nsim=2**4, batchsize=batchsize, pars_function=pars_function)
    funcs2 = define_functions(ecological_data, context_data, Nsim=2**7, batchsize=batchsize, pars_function=pars_function)

    if not try_gaussian:
        step2 = find_MAP(funcs1["dens_exact"], funcs1["sizes"], optax.adam(learning_rate=step2_lr), funcs2["dens_exact"], 
                                    Nmax=step2_Nmax, initial_guess=mu, Nmax2=step2_Nmax2)
    else:
        step2 = find_MAP(funcs1["dens_approx"], funcs1["sizes"], optax.adam(learning_rate=step2_lr), funcs2["dens_approx"], 
                                    Nmax=step2_Nmax, initial_guess=mu, Nmax2=step2_Nmax2)
        funcs1["dens_exact"] = funcs1["dens_approx"]

    mu = step2["pars"][-50:,].mean(0)
    est_std2 = jnp.array([jax.vmap(funcs2["dens_exact"])(mu[None,].repeat(5, 0), jax.random.split(jax.random.key(i), 5)) for i in range(20)]).flatten().std()

    # Step 3: Obtaining the full posterior distribution
    
    print("Step 3")

    if step3_type == "IS" or step3_type == "both":
        step3 = posterior_laplaceIS(mu, funcs2["dens_exact"], funcs2["sizes"], type_model, Nmax=step3_Nmax, Nchains=step3_Nchains, 
                            parallel=step3_parallel, dens_func_hess=funcs1["dens_exact"], mult=1.2)
        
        if step3_type == "both" and step3["ESS"]<200:
            step3 = posterior_RW(mu, funcs2["dens_exact"], funcs2["sizes"], type_model, Nmax=step3_Nmax, Nchains=step3_Nchains, 
                                 parallel=step3_parallel, dens_func_hess=funcs1["dens_exact"])

    elif step3_type == "RW":
        step3 = posterior_RW(mu, funcs2["dens_exact"], funcs2["sizes"], type_model, Nmax=step3_Nmax, Nchains=step3_Nchains, 
                             parallel=step3_parallel, dens_func_hess=funcs1["dens_exact"])
    
    else:
        step3 = None
        
    if (step3 is not None) and (not save_full_posterior): 
        step3["posterior"] = None
        step3["weight"] = None

    if try_gaussian:
        print("Step 3b")
        step3b = posterior_laplaceIS(mu, funcs2["dens_approx"], funcs2["sizes"], type_model, Nmax=step3_Nmax, Nchains=step3_Nchains, 
                            parallel=step3_parallel, dens_func_hess=funcs2["dens_approx"], mult=1.2)
        result["step3b"] = step3b
        
    result["step1"] = step1 
    result["step2"] = step2 
    result["step3"] = step3
    result["est_std"] = est_std2  
        
    with open(save_as, 'wb') as file: 
        pickle.dump(result, file)

    return result