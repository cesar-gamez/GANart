from keras.layers import Input, Reshape, Dropout, Dense, Flatten, BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam
import numpy as np
from PIL import Image
import os

#Generated image frame
ROWS = 4
COLS = 7
MARGIN = 4
SAVE_INTERVAL = 100

#Size for generating painting
NOISE_SIZE = 100

#Config
EPOCHS_AMOUNT = 10000
BATCH_SIZE = 10

GENERATE_RES = 3
IMG_SIZE = 128
IMG_CHANNELS = 3

training_data = np.load('resources/uploads/monet_data.npy')

def discriminator(image_shape):
    mod = Sequential()

    mod.add(Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, padding="same"))
    mod.add(LeakyReLU(alpha=0.2))
    mod.add(Dropout(0.25))

    mod.add(Conv2D(64, kernel_size=3, strides=2,padding="same"))
    mod.add(ZeroPadding2D(padding=((0,1), (0,1))))
    mod.add(BatchNormalization(momentum=0.8))
    mod.add(LeakyReLU(alpha=0.2))
    mod.add(Dropout(0.25))

    mod.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    mod.add(BatchNormalization(momentum=0.8))
    mod.add(LeakyReLU(alpha=0.2))
    mod.add(Dropout(0.25))

    mod.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    mod.add(BatchNormalization(momentum=0.8))
    mod.add(LeakyReLU(alpha=0.2))

    mod.add(Dropout(0.25))
    mod.add(Conv2D(512, kernel_size=3, strides=1, padding="same"))
    mod.add(BatchNormalization(momentum=0.8))
    mod.add(LeakyReLU(alpha=0.2))

    mod.add(Dropout(0.25))
    mod.add(Flatten())
    mod.add(Dense(1, activation='sigmoid'))

    input_image = Input(shape=image_shape)
    validity = mod(input_image)
    return Model(input_image, validity)

def generator(noise_size, channels):
    mod = Sequential()
    mod.add(Dense(4 * 4 * 256, activation="relu", input_dim=noise_size))
    mod.add(Reshape((4, 4, 256)))

    mod.add(UpSampling2D())
    mod.add(Conv2D(256, kernel_size=3, padding="same"))
    mod.add(BatchNormalization(momentum=0.8))
    mod.add(Activation("relu"))

    mod.add(UpSampling2D())
    mod.add(Conv2D(256, kernel_size=3, padding="same"))
    mod.add(BatchNormalization(momentum=0.8))
    mod.add(Activation("relu"))

    for i in range(GENERATE_RES):
        mod.add(UpSampling2D())
        mod.add(Conv2D(256, kernel_size=3, padding="same"))
        mod.add(BatchNormalization(momentum=0.8))
        mod.add(Activation("relu"))

    mod.summary()
    mod.add(Conv2D(channels, kernel_size=3, padding="same"))
    mod.add(Activation("tanh"))

    input = Input(shape=(noise_size))
    generated_image = mod(input)

    return Model(input, generated_image)

def save(cnt, noise):
    image_array = np.full((MARGIN + (ROWS * (IMG_SIZE + MARGIN)), MARGIN + (COLS * (IMG_SIZE + MARGIN)), 3), 255, dtype=np.uint8)

    generated_images = gen.predict(noise)

    generated_images = 0.5 * generated_images + 0.5

    image_count = 0
    for row in range(ROWS):
        for col in range(COLS):
            r = row * (IMG_SIZE + MARGIN) + MARGIN
            c = col * (IMG_SIZE + MARGIN) + MARGIN
            image_array[r:r + IMG_SIZE, c:c+IMG_SIZE] = generated_images[image_count] * 255
            image_count += 1

    output_path = 'output'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    filename = os.path.join(output_path, f"trained-{cnt}.png")
    im = Image.fromarray(image_array)
    im.save(filename)

image_shape = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)
optim = Adam(1.5e-4, 0.5)

discrim = discriminator(image_shape)
discrim.compile(loss="binary_crossentropy", optimizer=optim, metrics=["accuracy"])
gen = generator(NOISE_SIZE, IMG_CHANNELS)

random_input = Input(shape=(NOISE_SIZE))
generated_image = gen(random_input)
discrim.trainable = False
validity = discrim(generated_image)

combined = Model(random_input, validity)
combined.compile(loss="binary_crossentropy", optimizer=optim, metrics=["accuracy"])

y_real = np.ones((BATCH_SIZE, 1))
y_fake = np.zeros((BATCH_SIZE, 1))

fixed_noise = np.random.normal(0, 1, (ROWS * COLS, NOISE_SIZE))

cnt = 1
for epoch in range(EPOCHS_AMOUNT):
    idx = np.random.randint(0, training_data.shape[0], BATCH_SIZE)
    x_real = training_data[idx]

    noise = np.random.normal(0, 1, (BATCH_SIZE, NOISE_SIZE))
    x_fake = gen.predict(noise)

    discriminator_metric_real = discrim.train_on_batch(x_real, y_real)
    discriminator_metric_generated = discrim.train_on_batch(x_fake, y_fake)

    discriminator_metric = 0.5 * np.add(discriminator_metric_real, discriminator_metric_generated)

    generator_metric = combined.train_on_batch(noise, y_real)

    if epoch % SAVE_INTERVAL == 0:
        save(cnt, fixed_noise)
        cnt += 1

    print(f"{epoch} epoch, Discriminator accuracy: {100 * discriminator_metric[1]}, Generator accuracy: {100 * generator_metric[1]}")
