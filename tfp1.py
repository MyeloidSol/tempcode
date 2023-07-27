
# Simulate Fake Data ----
N = 100
height = tf.convert_to_tensor(np.random.normal(loc = 172, scale = 25, size = N))

# Iterations per-chain
Nsamp = 1000 # Number of samples per chain
Nburn = 1000 # Number of burn ins
Nchains = 4

def log_prob(mu, sigma, heights):
    target = 0

    # Priors
    mu_prior = tfd.Normal(150.0, 50.0)
    sigma_prior = tfd.Exponential(0.1)

    # Likelihood
    likelihood = tfd.Normal(mu, sigma)

    # Log-Probability for current set of parameters(mu and sigma)
    target += mu_prior.log_prob(mu)
    target += sigma_prior.log_prob(sigma)
    target += tf.reduce_sum(likelihood.log_prob(heights))

    return target


# HMC Kernel
hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=log_prob,
        step_size=0.01,
        num_leapfrog_steps=5
)

# This adapts the inner kernel's step_size.
adaptive_kernel = tfp.mcmc.SimpleStepSizeAdaptation(
  inner_kernel = hmc_kernel,
  num_adaptation_steps=int(Nburn * 0.8)
)

# Initial state for MCMC chains
initial_state = [150.0, 1.0]


samples, _ = tfp.mcmc.sample_chain(
    num_results=1,
    num_burnin_steps=1000,  # Number of burn-in steps (discard at the beginning)
    current_state=initial_state,
    kernel=adaptive_kernel
)

mu_samples, sigma_samples = samples
