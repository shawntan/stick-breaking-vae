import cPickle as pickle
import theano
import theano.tensor as T
import numpy as np
import math
import random
import vae
import omniglot_model
import feedforward
from theano_toolkit.parameters import Parameters
from theano_toolkit import updates
from pprint import pprint

def epoch_iterator(concept_count, example_count,
                   batch_size=10,
                   example_size=10):
    concept_idxs = np.arange(concept_count).astype(np.int32)
    example_idxs = np.arange(example_count).astype(np.int32)
    np.random.shuffle(concept_idxs)
    iterations = int(math.ceil(concept_count / float(batch_size)))
    example_batches = int(math.ceil(example_count / float(example_size)))

    train_idxs = []
    for i in xrange(iterations):
        np.random.shuffle(example_idxs)
        for j in xrange(example_batches):
            train_idxs.append((concept_idxs[i * batch_size:
                                            (i + 1) * batch_size],
                               example_idxs[j * example_size:
                                            (j + 1) * example_size]))
    random.shuffle(train_idxs)
    for x in train_idxs:
        yield x


def bernoulli_nll(X, mean):
    return -T.switch(T.eq(X, 1), T.log(mean), T.log(1 - mean))


def prepare_functions(
        input_size, hidden_size,
        concept_latent_size,
        style_latent_size,
        grad_mag,
        dataset):

    P = Parameters()
    reconstruct = omniglot_model.build(
        P, input_size, hidden_size,
        concept_latent_size, style_latent_size)

    concept_idx = T.ivector('concept_idx')
    style_idx = T.ivector('style_idx')

    X = T.tensor4('X')
    concept_mean, concept_std, \
        style_mean, style_std, X_recon_mean = reconstruct(X)

    concept_loss = T.mean(vae.kl_divergence(concept_mean, concept_std, 0, 1),
                          axis=-1)
    style_loss = T.mean(T.sum(vae.kl_divergence(style_mean, style_std, 0, 1),
                              axis=-1),
                        axis=-1)
    recon_loss = T.mean(T.sum(bernoulli_nll(X, X_recon_mean),
                              axis=(-3, -2, -1)),
                        axis=-1)

    parameters = P.values()
    reg_loss = concept_loss + style_loss
    loss = recon_loss + reg_loss
    cost = (loss +
            5e-4 * sum(T.sum(T.sqr(w)) for w in parameters
                       if w.name.startswith('W') or
                       w not in (P.W_decoder_input_input_0,
                                 P.W_decoder_input_input_1))) / X.shape[1]
#    cost = loss / X.shape[1]

    gradients = updates.clip_deltas(T.grad(cost, wrt=parameters), grad_mag)

    lr = T.scalar('lr')
    P_train = Parameters()
    X_shared = theano.shared(dataset)
    train = theano.function(
        inputs=[concept_idx, style_idx, lr],
        outputs=[loss / X.shape[1],
                 recon_loss / X.shape[1],
                 concept_loss,
                 style_loss / X.shape[1]],
        updates=updates.adam(parameters, gradients,
                             learning_rate=lr, P=P_train),
        givens={X: T.cast(X_shared[concept_idx.dimshuffle(0, 'x'),
                                   style_idx.dimshuffle('x', 0)], 'float32')})

    test = theano.function(
        inputs=[X],
        outputs=[loss / X.shape[1],
                 recon_loss / X.shape[1],
                 concept_loss,
                 style_loss / X.shape[1],
                 X_recon_mean])

    return train, test, P, P_train

if __name__ == "__main__":
    import sys
    import omniglot
    data_file = sys.argv[1]
    model_file = sys.argv[2]
    batch_size = int(sys.argv[3])
    learning_rate = float(sys.argv[4])
    grad_mag = float(sys.argv[5])
    log_file = open(sys.argv[6], 'w', 0)

    data, validation_data = omniglot.load(data_file, size=(32, 32),
                                          validation=0.1)
    print "Data shape:", data.shape
    train, test, P, P_train = prepare_functions(
        input_size=data.shape[-1],
        hidden_size=512,
        concept_latent_size=32,
        style_latent_size=16,
        grad_mag=grad_mag,
        dataset=data)

    if False:
        P.load(model_file)
        P_train.load(model_file + ".trn")
        print "Loaded model."
    else:
        shape = P.W_decoder_input_output.get_value().shape
        P.W_decoder_input_output.set_value(
            feedforward.relu_init(*shape))
        P.W_deconv_stack_2.set_value(
            P.W_deconv_stack_2.get_value() * 0.)
        P.W_decoder_input_input_0.set_value(
            P.W_decoder_input_input_0.get_value() * 1)
        P.W_decoder_input_input_1.set_value(
            P.W_decoder_input_input_1.get_value() * 10)
        P.W_style_inferer_mean
        P.b_style_inferer_mean
        P.W_style_inferer_std
        P.b_style_inferer_std
    print "train set size:", data.shape[0]
    print "test set size:", validation_data.shape[0]
    score, _, _, _, sample = test(validation_data)
    save_figure(validation_data, sample)
    best_score = score
    print "Parameters to tune:"
    pprint(P.values())

    def run_training(epochs, concept_count, example_count,
                     batch_size, learning_rate):
        global best_score
        for epoch in xrange(epochs):
            for c, s in epoch_iterator(data.shape[0],
                                       example_count, batch_size):
                train(c, s, learning_rate)
                # print ' '.join(str(v) for v in train(c, s, learning_rate))
            score, recon_loss, concept_loss, \
                style_loss, sample = test(validation_data)
            print "Validation loss:", score,\
                "recon_loss:", recon_loss,\
                "reg_loss: (", concept_loss, ", ", style_loss, ")",\
                "lr:", learning_rate,\
                "batch_size:", batch_size,
            print >> log_file, score
            if score < best_score:
                best_score = score
                P.save(model_file)
                P_train.save(model_file + ".trn")
                save_figure(validation_data, sample)
                print "Saved."
            else:
                print

    print "Starting training..."
    run_training(200, data.shape[0], data.shape[1],
                 batch_size=batch_size, learning_rate=learning_rate)
