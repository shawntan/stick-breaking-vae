import theano.tensor as T
from theano_toolkit import utils as U
from theano_toolkit import hinton


def beta_sample(alpha):
    u = U.theano_rng.uniform(size=alpha.shape)
    return 1 - u ** (1 / alpha)

def beta_kl_divergence(alpha):
    # Divergence from Beta(1, 1)
    b = alpha
    return T.log(b) + (1 - b) / b

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
    return sample_pi
