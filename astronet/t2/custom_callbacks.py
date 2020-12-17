import imageio
import io
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from IPython.display import Image as IPyImage
from PIL import Image

# Visualization utilities
plt.rc('font', size=20)
plt.rc('figure', figsize=(15, 3))

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


class DetectOverfittingCallback(tf.keras.callbacks.Callback):
    def __init__(self, threshold=0.7):
        super(DetectOverfittingCallback, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        ratio = logs["val_loss"] / logs["loss"]
        print("Epoch: {}, Val/Train loss ratio: {:.2f}".format(epoch, ratio))

        if ratio > self.threshold:
            print("Stopping training...")
            self.model.stop_training = True


class VisCallback(tf.keras.callbacks.Callback):
    def __init__(self, inputs, ground_truth, display_freq=10, n_samples=10):
        self.inputs = inputs
        self.ground_truth = ground_truth
        self.images = []
        self.display_freq = display_freq
        self.n_samples = n_samples

    def __display_digits(inputs, outputs, ground_truth, epoch, n=10):
        plt.clf()

        plt.yticks([])
        plt.grid(None)
        inputs = np.reshape(inputs, [n, 28, 28])
        inputs = np.swapaxes(inputs, 0, 1)
        inputs = np.reshape(inputs, [28, 28 * n])
        plt.imshow(inputs)
        plt.xticks([28 * x + 14 for x in range(n)], outputs)
        for i, t in enumerate(plt.gca().xaxis.get_ticklabels()):
            if outputs[i] == ground_truth[i]:
                t.set_color("green")
            else:
                t.set_color("red")
        plt.grid(None)

    def on_epoch_end(self, epoch, logs=None):
        # Randomly sample data
        np.random.seed(RANDOM_SEED)
        indexes = np.random.choice(len(self.inputs), size=self.n_samples)
        X_test, y_test = self.inputs[indexes], self.ground_truth[indexes]
        predictions = np.argmax(self.model.predict(X_test), axis=1)

        # Plot the digits
        self.__display_digits(X_test, predictions, y_test, epoch, n=self.display_freq)

        # Save the figure
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf)
        self.images.append(np.array(image))

        # Display the digits every 'display_freq' number of epochs
        if epoch % self.display_freq == 0:
            plt.show()

    def on_train_end(self, logs=None):
        GIF_PATH = './animation.gif'
        imageio.mimsave(GIF_PATH, self.images, fps=1)