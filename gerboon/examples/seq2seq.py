import numpy as np
import keras as ks
import keras.backend as K
import matplotlib.pyplot as plt

batch_size = 128

seq_len_in = 100
seq_len_out = 100


def generate_sines(start_angle, length):
    xs = np.linspace(start_angle, start_angle + length / 15, length)
    s1 = np.sin(xs)
    s2 = np.sin((xs + 1) * 2.3)
    sout = s1 * s2
    return np.stack((s1, s2), axis=1), np.expand_dims(sout, axis=1)


def gen_batch():
    while True:
        batch_xe = []
        batch_xd = []
        batch_y = []
        for i in range(batch_size):
            # Generate a random starting_angle
            sa = np.random.uniform(-5 * np.pi * 2, 5 * np.pi * 2, size=None)
            x, y = generate_sines(sa, seq_len_in + seq_len_out)

            # Generate inputs and outputs
            xdec = y[seq_len_in - 1:-1]
            y = y[seq_len_in:]

            # Append the samples to the batches
            batch_xe.append(x[:seq_len_in])
            batch_xd.append(xdec)
            batch_y.append(y)

        batch_xe = np.stack(batch_xe)
        batch_xd = np.stack(batch_xd)
        batch_y = np.stack(batch_y)
        yield [batch_xe, batch_xd], batch_y


def build_seq2seq_model(use_noise=False):
    # Define model inputs for the encoder/decoder stack
    x_enc = ks.Input(shape=(None, 2))
    x_dec = ks.Input(shape=(None, 1), name="x_dec")

    # This is not normally in seq2seq models (from what I know).
    # During training the model gets the input for the previous timestep.
    # Therefore it can be easy to just rely on this value for the most part, since it's always correct
    # During prediction however, it isn't the real output, but the predicted.
    # This creates problems when predicting over large time horizons, since all small errors add up
    # To train the network to rely more on it's memory, I add noise to the inputs during training
    if use_noise:
        x_dec_t = ks.layers.GaussianNoise(0.2)(x_dec)
    else:
        x_dec_t = x_dec


    # Define the encoder GRU, which only has to return a state
    _, state = ks.layers.GRU(40, return_state=True)(x_enc)

    # Define the decoder GRU and the Dense layer that will transform sequences of size 20 vectors to
    # a sequence of 1-long vectors of final predicted values
    dec_gru = ks.layers.GRU(40, return_state=True, return_sequences=True)
    dec_dense = ks.layers.TimeDistributed(ks.layers.Dense(1, activation='linear'))

    # Use these definitions to calculate the outputs of out encoder/decoder stack
    dec_intermediates, _ = dec_gru(x_dec_t, initial_state=state)
    dec_outs = dec_dense(dec_intermediates)

    # Define the encoder/decoder stack model
    encdecmodel = ks.Model(inputs=[x_enc, x_dec], outputs=dec_outs)

    # Define the encoder model
    E = ks.Model(inputs=x_enc, outputs=state)

    # Define a state_in model for the Decoder model (which will be used for prediction)
    state_in = ks.Input(shape=(40,), name="state")

    # Use the previously defined layers to calculate the new output value and state for the prediction model as well
    dec_intermediate, new_state = dec_gru(x_dec, initial_state=state_in)
    dec_out = dec_dense(dec_intermediate)

    # Define the decoder/prediction model
    D = ks.Model(inputs=[x_dec, state_in], outputs=[dec_out, new_state])
    return E, D, encdecmodel


def make_prediction(E, D, previous_timesteps_x, previous_y, n_output_timesteps):
    # Get the state from the Encoder using the previous timesteps for x
    state = E.predict(np.expand_dims(previous_timesteps_x, axis=0))

    # Initialize the outputs on the previous y so we have something to feed the net
    # It might be neater to feed a start symbol instead
    outp = np.expand_dims(previous_y, axis=0)
    outputs = []
    for i in range(n_output_timesteps):
        outp, state = D.predict([outp, state])
        outputs.append(outp)
    return np.concatenate(outputs, axis=1)[0]


if __name__ == "__main__":
    encoder, decoder, encdecmodel = build_seq2seq_model(use_noise=True)

    encdecmodel.compile(ks.optimizers.Adam(0.03), ks.losses.mean_squared_error)
    encdecmodel.fit_generator(gen_batch(), steps_per_epoch=100, epochs=3)

    x, y = generate_sines(0.5, 200)

    # plt.plot(x[:, 0])
    # plt.plot(x[:, 1])
    plt.plot(y)

    predictions = make_prediction(encoder, decoder, x[:100], y[99:100], 100)
    plt.plot(np.arange(100, 200), predictions)
    plt.show()
