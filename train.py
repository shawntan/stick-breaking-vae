if __name__ == "__main__":
    from theano_toolkit import hinton
    from theano_toolkit.parameters import Parameters
    import numpy as np
    P = Parameters()
    encode_decode = build(P, 20, 40, 10, 10)

    X = T.as_tensor_variable(np.random.randn(32, 20).astype(np.float32))
    z_means, z_stds, alphas, X_mean, log_pi_samples = encode_decode(X)

    y = (recon_loss(X, X_mean, log_pi_samples)).eval()
    print y.shape
