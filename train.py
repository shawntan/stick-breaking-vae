import theano
import theano.tensor as T
import numpy as np
import math
import random
from theano_toolkit.parameters import Parameters
from theano_toolkit import updates
from pprint import pprint

import data
import model


def load_data_frames(filename):
    (data_train_X, _), \
        (data_valid_X, _), _ = data.load('data/mnist.pkl.gz')

    train_X = theano.shared(data_train_X)
    valid_X = theano.shared(data_valid_X)
    return train_X, valid_X


def prepare_functions(input_size, hidden_size, latent_size, step_count,
                      batch_size, train_X, valid_X):
    P = Parameters()
    encode_decode = model.build(P,
                                input_size=input_size,
                                hidden_size=hidden_size,
                                latent_size=latent_size,
                                step_count=step_count)
    P.W_decoder_input_0.set_value(
        P.W_decoder_input_0.get_value() * 10)

    X = T.matrix('X')
    Z_means, Z_stds, alphas, X_mean, log_pi_samples = encode_decode(X)
    recon_loss = T.mean(model.recon_loss(X, X_mean, log_pi_samples), axis=0)
    reg_loss = T.mean(model.reg_loss(Z_means, Z_stds, alphas), axis=0)
    vlb = recon_loss + reg_loss

    parameters = P.values()
    cost = vlb + 1e-4 * sum(T.sum(T.sqr(w))
                            for w in parameters)
    gradients = updates.clip_deltas(T.grad(cost, wrt=parameters), 5)

    print "Updated parameters:"
    pprint(parameters)
    idx = T.iscalar('idx')

    train = theano.function(
        inputs=[idx],
        outputs=[vlb, recon_loss, reg_loss,
                 T.max(T.argmax(log_pi_samples, axis=0))],
        updates=updates.adam(parameters, gradients,
                             learning_rate=1e-4),
        givens={X: train_X[idx * batch_size: (idx + 1) * batch_size]}
    )

    validate = theano.function(
        inputs=[],
        outputs=[vlb, recon_loss, reg_loss],
        givens={X: valid_X}
    )

    return train, validate

if __name__ == "__main__":
    epochs = 100
    batch_size = 32
    print "Loading data..."
    train_X, valid_X = load_data_frames('data/mnist.pkl.gz')
    train_X_data = train_X.get_value()
    print "Compiling functions..."
    train, validate = prepare_functions(input_size=train_X_data.shape[1],
                                        hidden_size=512,
                                        latent_size=20,
                                        step_count=10,
                                        batch_size=batch_size,
                                        train_X=train_X,
                                        valid_X=valid_X)

    batches = int(math.ceil(train_X_data.shape[0] / float(batch_size)))
    print "Starting training..."
    for epoch in xrange(epochs):
        np.random.shuffle(train_X_data)
        train_X.set_value(train_X_data)
        for i in xrange(batches):
            vals = train(i)
            print ' '.join(map(str, vals))
