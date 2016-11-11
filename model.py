import theano
import theano.tensor as T
import numpy as np

import lstm
import vae
import feedforward
import stick_break_vae

from theano_toolkit import utils as U


def build(P, input_size, hidden_size, latent_size, step_count):
    encode = build_encoder(P, input_size, hidden_size, latent_size, step_count)
    decode = build_decoder(P,
                           latent_size=latent_size,
                           hidden_size=hidden_size,
                           output_size=input_size)

    sample_log_pi = stick_break_vae.build_sample_pi(size=step_count)

    def encode_decode(X):
        z_samples, z_means, z_stds, alphas = encode(X)
        log_pi_samples = sample_log_pi(alphas.T).T
        X_recon = decode(z_samples)
        return z_means, z_stds, alphas, X_recon, log_pi_samples
    return encode_decode


def build_encoder(P, input_size, hidden_size, latent_size, step_count):

    P.init_encoder_hidden = np.zeros((hidden_size,))
    P.init_encoder_cell = np.zeros((hidden_size,))

    P.w_encoder_v = np.zeros((hidden_size,))
    P.b_encoder_v = 0

    rnn_step = lstm.build_step(P,
                               name="encoder",
                               input_sizes=[input_size, latent_size],
                               hidden_size=hidden_size)

    gaussian_out = vae.build_encoder_output(
            P, name="encoder_gaussian",
            input_size=hidden_size,
            output_size=latent_size,
            initialise_weights=None)

    def encode(X):
        init_hidden = T.tanh(P.init_encoder_hidden)
        init_cell = P.init_encoder_cell
        init_hidden_batch = T.alloc(init_hidden, X.shape[0], hidden_size)
        init_cell_batch = T.alloc(init_cell, X.shape[0], hidden_size)
        init_latent = U.theano_rng.normal(size=(X.shape[0], latent_size))
        eps_seq = U.theano_rng.normal(size=(step_count,
                                            X.shape[0],
                                            latent_size))

        def step(eps, prev_latent, prev_hidden, prev_cell):
            hidden, cell = rnn_step(X,
                                    prev_latent,
                                    prev_hidden,
                                    prev_cell)
            _, z_mean, z_std = gaussian_out(hidden)
            z_sample = z_mean + eps * z_std
            return z_sample, hidden, cell, z_mean, z_std

        [z_samples, hiddens, cells, z_means, z_stds], _ = theano.scan(
            step,
            sequences=[eps_seq],
            outputs_info=[init_latent,
                          init_hidden_batch,
                          init_cell_batch,
                          None,
                          None]
        )

        alphas = T.exp(T.dot(hiddens, P.w_encoder_v) + P.b_encoder_v)
        return z_samples, z_means, z_stds, alphas

    return encode


def build_decoder(P, latent_size, hidden_size, output_size):
    decode_ = feedforward.build_classifier(
        P, name='decoder',
        input_sizes=[latent_size],
        hidden_sizes=[hidden_size],
        output_size=output_size,
        initial_weights=feedforward.relu_init,
        activation=T.nnet.softplus,
        output_activation=T.nnet.sigmoid)

    def decode(X):
        return decode_([X])[1]

    return decode


def reg_loss(z_means, z_stds, alphas):
    gaussian_loss = T.sum(
            vae.kl_divergence(z_means, z_stds, 0, 1), axis=0)
    stick_break_loss = T.sum(
            stick_break_vae.kl_divergence(alphas), axis=0)
    return gaussian_loss + stick_break_loss


def recon_loss(X, X_mean, log_pi_samples):
    log_p = T.sum(X * T.log(X_mean) + (1 - X) * T.log(1 - X_mean), axis=-1)
    log_pi_t = log_p + log_pi_samples
    k = T.max(log_pi_t, axis=0)
    return -(T.log(T.sum(T.exp(log_pi_t - k), axis=0)) + k)
