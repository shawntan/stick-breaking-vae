import theano
import theano.tensor as T
import numpy as np
import math
from theano_toolkit.parameters import Parameters
from theano_toolkit import updates
from pprint import pprint
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import data
import model


def plot_samples(x, x_samples, max_component):
    plt.figure(figsize=(20, 20))
    for i in xrange(10):
        ax = plt.subplot2grid((13, 13), (i, 0))
        ax.imshow(x[i].reshape(28, 28), cmap='Greys', interpolation='None')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.axis('off')

        for j in xrange(10):
            ax = plt.subplot2grid((13, 13), (i, j + 2))
            ax.imshow(x_samples[j, i].reshape(28, 28),
                      cmap='Greys',
                      interpolation='None')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            if j == max_component[i]:
                ax.spines['bottom'].set_color('red')
                ax.spines['top'].set_color('red')
                ax.spines['right'].set_color('red')
                ax.spines['left'].set_color('red')
            else:

                ax.axis('off')
    plt.savefig('sample.png', bbox_inches='tight')
    plt.close()


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
                                latent_size=latent_size)
    P.W_decoder_input_0.set_value(
        P.W_decoder_input_0.get_value() * 10)

    X = T.matrix('X')
    step_count = T.iscalar('step_count')
    Z_means, Z_stds, alphas, \
        X_mean, log_pi_samples = encode_decode(X, step_count=step_count)
    recon_loss = T.mean(model.recon_loss(X, X_mean, log_pi_samples), axis=0)
    reg_loss = T.mean(model.reg_loss(Z_means, Z_stds, alphas), axis=0)
    vlb = recon_loss + reg_loss

    parameters = P.values()
    cost = vlb + 1e-3 * sum(T.sum(T.sqr(w))
                            for w in parameters)
    gradients = updates.clip_deltas(T.grad(cost, wrt=parameters), 5)

    print "Updated parameters:"
    pprint(parameters)
    idx = T.iscalar('idx')

    train = theano.function(
        inputs=[idx, step_count],
        outputs=[vlb, recon_loss, reg_loss,
                 T.max(T.argmax(log_pi_samples, axis=0))],
        updates=updates.adam(parameters, gradients,
                             learning_rate=1e-4),
        givens={X: train_X[idx * batch_size: (idx + 1) * batch_size]}
    )

    validate = theano.function(
        inputs=[],
        outputs=vlb,
        givens={X: valid_X, step_count: np.int32(10)}
    )

    sample = theano.function(
        inputs=[],
        outputs=[X, X_mean, T.argmax(log_pi_samples, axis=0)],
        givens={X: valid_X[:10], step_count: np.int32(10)}
    )

    return train, validate, sample

if __name__ == "__main__":
    epochs = 100
    batch_size = 32
    print "Loading data..."
    train_X, valid_X = load_data_frames('data/mnist.pkl.gz')
    train_X_data = train_X.get_value()
    print "Compiling functions..."
    train, validate, sample = prepare_functions(
        input_size=train_X_data.shape[1],
        hidden_size=64,
        latent_size=16,
        step_count=10,
        batch_size=batch_size,
        train_X=train_X,
        valid_X=valid_X)

    batches = int(math.ceil(train_X_data.shape[0] / float(batch_size)))
    print "Starting training..."
    best_score = np.inf
    for epoch in xrange(epochs):
        vlb = validate()
        print vlb,
        if vlb < best_score:
            x, x_samples, max_component = sample()
            plot_samples(x, x_samples, max_component)
            best_score = vlb
            print "Saved."
        else:
            print

        np.random.shuffle(train_X_data)
        train_X.set_value(train_X_data)
        for i in xrange(batches):
            steps = np.random.randint(1, 11)
            vals = train(i, 10)
            print ' '.join(map(str, vals))
