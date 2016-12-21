import theano.tensor as T
from model import recon_loss, sharpen
import stick_break_vae
import vae


def estimated_marginal_likelihood(X, encode, decode,
                                  step_count, sample_count):
    sample_log_pi = stick_break_vae.build_sample_pi_(size=step_count)
    _X = X
    X = T.repeat(X, sample_count, axis=0)
    z_samples, z_means, z_stds, alphas = encode(X, step_count)
    sample_indiv, log_probs = sample_log_pi(alphas.T)
    sample_indiv = sample_indiv.T
    log_pi_samples = sharpen(log_probs.T, 1)
    X_mean = decode(z_samples)
    log_q_z_given_x = T.sum(beta_prob(sample_indiv, alphas) -
                            vae.gaussian_nll(z_samples, z_means, z_stds),
                            axis=0)

    log_p_z = T.sum(beta_prob(sample_indiv, 1) -
                    vae.gaussian_nll(z_samples, 0, 1),
                    axis=0)
    log_p_x_given_z = -recon_loss(X, X_mean, log_pi_samples)[0]
    log_p_x_ests = log_p_z + log_p_x_given_z - log_q_z_given_x
    log_p_x_ests = log_p_x_ests.reshape((_X.shape[0], sample_count))

    k = T.max(log_p_x_ests, axis=1, keepdims=True)
    norm_p_x_ests = T.mean(T.exp(log_p_x_ests - k), axis=1)
    log_p_x_approx = T.log(norm_p_x_ests) + k[:, 0]

    return T.mean(log_p_x_approx)


def beta_prob(v, alpha):
    # Probability of v given Beta(1, alpha)
    return T.log(alpha) + (alpha - 1) * T.log(1 - v)
