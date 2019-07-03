import os
import numpy as np
import matplotlib.pyplot as plt

use_cpu = True
latent_size = 8
h_size = 256
batch_size = 64
lr = 0.0002


if use_cpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import keras as ks
X = ks.datasets.mnist.load_data()[0][0].astype(np.float32)
X /= 255.0
print(X.min(), X.max())

generator = ks.models.Sequential()
generator.add(ks.layers.Dense(h_size, activation='relu', input_shape=(latent_size,)))
generator.add(ks.layers.BatchNormalization())
generator.add(ks.layers.ReLU())
generator.add(ks.layers.Dense(28*28, activation='sigmoid'))
generator.add(ks.layers.Reshape((28, 28)))

discriminator = ks.models.Sequential()
discriminator.add(ks.layers.Reshape((28*28,), input_shape=(28, 28)))
discriminator.add(ks.layers.Dense(h_size))
discriminator.add(ks.layers.LeakyReLU(0.2))
discriminator.add(ks.layers.Dense(1, activation='sigmoid'))
discriminator.compile(ks.optimizers.RMSprop(lr), ks.losses.binary_crossentropy)

gan_stack = ks.models.Sequential()
gan_stack.add(generator)
discriminator.trainable = False
gan_stack.add(discriminator)
gan_stack.compile(ks.optimizers.RMSprop(lr), ks.losses.binary_crossentropy)


test_zs = np.random.normal(0, 1, (10, latent_size))

# Training loop
for step in range(10000):
    # ======= TRAIN D =======
    # sample real xs
    indices = np.random.choice(np.arange(0, X.shape[0]), batch_size//2)
    real_xs = X[indices]

    # Sample zs to generate fake samples
    zs = np.random.normal(0, 1, (batch_size//2, latent_size))
    fake_xs = generator.predict(zs)

    mbatch_D = np.concatenate([real_xs, fake_xs], axis=0)
    ys = np.zeros((batch_size, 1))

    # Set real samples in ys
    ys[:batch_size//2] = 1.0
    discriminator.fit(mbatch_D, ys, batch_size=batch_size, verbose=False)

    # ======= TRAIN G =======
    zs = np.random.normal(0, 1, (batch_size, latent_size))
    ys = np.ones((batch_size, 1))

    gan_stack.fit(zs, ys, batch_size=batch_size, verbose=False)

    if step % 1000 == 0:
        # Show results
        generated_samples = generator.predict(test_zs)
        img = np.concatenate(list(generated_samples), axis=1)
        plt.imshow(img, cmap='gray')
        plt.show()



