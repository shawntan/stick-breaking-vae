import theano.tensor as T
from theano_toolkit import utils as U
from theano_toolkit import hinton

# Beta sampling via inverse CDF
def beta_sample(alpha):
    u = U.theano_rng.uniform(size=alpha.shape)
    return 1 - u ** (1 / alpha)

# KL-divergence between gaussians
def gauss_kl_divergence(mu, sigma_sq):
    # Divergence from standard gaussian
    k = mu.shape
    return 1/2*(T.nlinalg.trace(sigma_sq) + T.dot(mu, mu - k + T.log(T.det(sigma_sq))))

                                                 
def beta_kl_divergence(alpha):
    # Divergence from Beta(1, 1)
    b = alpha
    return T.log(b) + (1 - b) / b

# KL-divergence between betas 
def betas_kl_divergence(alpha1, alpha2, T):
    # Divergence between beta(1, alpha1) and beta(1, alpha2)
    # T is truncation of the power series
    pass
        
# gaussian part of the full kl_divergence
def gauss_kl_divergence_sum(mus, sigma_sqs):
    return T.sum(gauss_kl_divergence(mus, sigma_sqs), axis-=1)

# beta part of the full kl-divergence     
def kl_divergence(alphas):
    return T.sum(beta_kl_divergence(alphas),
                 axis=-1)

def build_sample_pi(size=20):
    idx = T.arange(size)
    cum_mat = (idx.dimshuffle(0, 'x') <
               idx.dimshuffle('x', 0))

    def sample_log_pi(alphas):
        sample_indiv = beta_sample(alphas)
        log_v = T.log(sample_indiv)
        log_v = T.set_subtensor(log_v[:, -1], 0)
        neg_log_v = T.log(1 - sample_indiv)
        log_probs = T.dot(neg_log_v, cum_mat) + log_v
        return log_probs
    return sample_log_pi

