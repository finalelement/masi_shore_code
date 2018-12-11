from scipy.io import loadmat, savemat
from keras.layers import Input, Dense, Dropout
from keras.models import Sequential
from keras.layers import Activation
from keras.callbacks import EarlyStopping, CSVLogger
import tensorflow as tf
from keras import backend as K
import os
import os.path


def acc_loss_6th(y_true, y_pred):
    # Subtract mean value from expansions of true and pred
    x_true = y_true[:, 1:28]
    x_pred = y_pred[:, 1:28]

    # Normalize each vector
    comp_true = tf.conj(x_true)
    norm_true = x_true / tf.sqrt(tf.reduce_sum(tf.multiply(x_true, comp_true)))

    comp_pred = tf.conj(x_pred)
    norm_pred = x_pred / tf.sqrt(tf.reduce_sum(tf.multiply(x_pred, comp_pred)))

    comp_p2 = tf.conj(norm_pred)
    acc = tf.real(tf.reduce_sum(tf.multiply(norm_true, comp_p2)))
    acc = -1.0 * acc * acc

    loss_mse = K.mean(K.square(y_pred - y_true))

    return acc

def acc_loss_8th(y_true, y_pred):
    # Subtract mean value from expansions of true and pred
    x_true = y_true[:, 1:45]
    x_pred = y_pred[:, 1:45]

    # Normalize each vector
    comp_true = tf.conj(x_true)
    norm_true = x_true / tf.sqrt(tf.reduce_sum(tf.multiply(x_true, comp_true)))

    comp_pred = tf.conj(x_pred)
    norm_pred = x_pred / tf.sqrt(tf.reduce_sum(tf.multiply(x_pred, comp_pred)))

    comp_p2 = tf.conj(norm_pred)
    acc = tf.real(tf.reduce_sum(tf.multiply(norm_true, comp_p2)))
    acc = -1.0 * acc * acc

    loss_mse = K.mean(K.square(y_pred - y_true))

    return acc + loss_mse

def build_base_network():
    model = Sequential()
    # Input layer with dimension 1 and hidden layer i with 128 neurons.
    model.add(Dense(50, input_shape=(50,)))
    model.add(Dense(400))
    model.add(Activation("relu"))
    # model.add(Dropout(0.6))
    # Hidden layer j with 64 neurons plus activation layer.
    model.add(Dense(200))
    model.add(Activation("relu"))
    # model.add(Dropout(0.5))
    # Hidden layer k with 64 neurons.
    model.add(Dense(66))
    model.add(Activation("relu"))
    # model.add(Dropout(0.5))
    model.add(Dense(200))
    model.add(Activation("relu"))
    # model.add(Dropout(0.5))
    model.add(Dense(400))
    # model.add(Activation("relu"))
    # Output Layer.
    model.add(Dense(50))

    # Model is derived and compiled using mean square error as loss
    # function, accuracy as metric and gradient descent optimizer.
    model.compile(loss='mse', optimizer='RMSProp', metrics=['mse', 'mae'])
    model.summary()
    return model


def build_base_network_sh_8th():
    model = Sequential()
    # Input layer with dimension 1 and hidden layer i with 128 neurons.
    model.add(Dense(50, input_shape=(50,)))
    model.add(Dense(400))
    model.add(Activation("relu"))
    # model.add(Dropout(0.6))
    # Hidden layer j with 64 neurons plus activation layer.
    model.add(Dense(200))
    model.add(Activation("relu"))
    # model.add(Dropout(0.5))
    # Hidden layer k with 64 neurons.
    model.add(Dense(66))
    model.add(Activation("relu"))
    # model.add(Dropout(0.5))
    model.add(Dense(200))
    model.add(Activation("relu"))
    # model.add(Dropout(0.5))
    model.add(Dense(400))
    # model.add(Activation("relu"))
    # Output Layer.
    model.add(Dense(45))

    # Model is derived and compiled using mean square error as loss
    # function, accuracy as metric and gradient descent optimizer.
    model.compile(loss='mse', optimizer='RMSProp', metrics=['mse', 'mae'])
    model.summary()
    return model

def build_base_network_sh_6th():
    model = Sequential()
    # Input layer with dimension 1 and hidden layer i with 128 neurons.
    model.add(Dense(50, input_shape=(50,)))
    model.add(Dense(400))
    model.add(Activation("relu"))
    # model.add(Dropout(0.6))
    # Hidden layer j with 64 neurons plus activation layer.
    model.add(Dense(200))
    model.add(Activation("relu"))
    # model.add(Dropout(0.5))
    # Hidden layer k with 64 neurons.
    model.add(Dense(66))
    model.add(Activation("relu"))
    # model.add(Dropout(0.5))
    model.add(Dense(200))
    model.add(Activation("relu"))
    # model.add(Dropout(0.5))
    model.add(Dense(400))
    # model.add(Activation("relu"))
    # Output Layer.
    model.add(Dense(28))

    # Model is derived and compiled using mean square error as loss
    # function, accuracy as metric and gradient descent optimizer.
    model.compile(loss='mse', optimizer='RMSProp', metrics=['mse', 'mae'])
    model.summary()
    return model



def build_base_network_sh_8th_acc():
    model = Sequential()
    # Input layer with dimension 1 and hidden layer i with 128 neurons.
    model.add(Dense(50, input_shape=(50,)))
    model.add(Dense(400))
    model.add(Activation("relu"))
    # model.add(Dropout(0.6))
    # Hidden layer j with 64 neurons plus activation layer.
    model.add(Dense(200))
    model.add(Activation("relu"))
    # model.add(Dropout(0.5))
    # Hidden layer k with 64 neurons.
    model.add(Dense(66))
    model.add(Activation("relu"))
    # model.add(Dropout(0.5))
    model.add(Dense(200))
    model.add(Activation("relu"))
    # model.add(Dropout(0.5))
    model.add(Dense(400))
    # model.add(Activation("relu"))
    # Output Layer.
    model.add(Dense(45))

    # Model is derived and compiled using mean square error as loss
    # function, accuracy as metric and gradient descent optimizer.
    model.compile(loss=acc_loss_8th, optimizer='RMSProp', metrics=['mse', 'mae'])
    model.summary()
    return model

def build_base_network_sh_6th_acc():
    model = Sequential()
    # Input layer with dimension 1 and hidden layer i with 128 neurons.
    model.add(Dense(50, input_shape=(50,)))
    model.add(Dense(400))
    model.add(Activation("relu"))
    # model.add(Dropout(0.6))
    # Hidden layer j with 64 neurons plus activation layer.
    model.add(Dense(200))
    model.add(Activation("relu"))
    # model.add(Dropout(0.5))
    # Hidden layer k with 64 neurons.
    model.add(Dense(66))
    model.add(Activation("relu"))
    # model.add(Dropout(0.5))
    model.add(Dense(200))
    model.add(Activation("relu"))
    # model.add(Dropout(0.5))
    model.add(Dense(400))
    # model.add(Activation("relu"))
    # Output Layer.
    model.add(Dense(28))

    # Model is derived and compiled using mean square error as loss
    # function, accuracy as metric and gradient descent optimizer.
    model.compile(loss=acc_loss_6th, optimizer='RMSProp', metrics=['mse', 'mae'])
    model.summary()
    return model

def build_base_network_sh_to_sh_8th_acc():
    model = Sequential()
    # Input layer with dimension 1 and hidden layer i with 128 neurons.
    model.add(Dense(45, input_shape=(45,)))
    model.add(Dense(400))
    model.add(Activation("relu"))
    # model.add(Dropout(0.6))
    # Hidden layer j with 64 neurons plus activation layer.
    model.add(Dense(200))
    model.add(Activation("relu"))
    # model.add(Dropout(0.5))
    # Hidden layer k with 64 neurons.
    model.add(Dense(66))
    model.add(Activation("relu"))
    # model.add(Dropout(0.5))
    model.add(Dense(200))
    model.add(Activation("relu"))
    # model.add(Dropout(0.5))
    model.add(Dense(400))
    # model.add(Activation("relu"))
    # Output Layer.
    model.add(Dense(45))

    # Model is derived and compiled using mean square error as loss
    # function, accuracy as metric and gradient descent optimizer.
    model.compile(loss=acc_loss_8th, optimizer='RMSProp', metrics=['mse', 'mae'])
    model.summary()
    return model


# Originally number of epochs was set to 1000, currently at 10.
def train_nn(model, X, y, out_dir, val_size=0.1, n_epoch=1000, batch_s=1000):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    csv_logger = CSVLogger(os.path.join(out_dir, 'results.csv'))

    model.fit(X, y, epochs=n_epoch, batch_size=batch_s, verbose=1, shuffle=True, validation_split=val_size, callbacks=[csv_logger])
    return model


def save_estimate(model, X, y, out_file):
    # y_pred = model.predict(X)

    # y_pred = y_scaler.inverse_transform(y_pred)
    # y = y_scaler.inverse_transform(y)

    out_path = os.path.dirname(out_file)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    y_pred = model.predict(X)
    savemat(out_file, mdict={'out_pred': y_pred, 'out_true': y})


def save_test_set_prediction(model, out_file, X_test):
    # Get dimensions of arrays
    x_size = X_test.shape
    print('Histology Test: Input Array Shape', x_size)

    # Make Predictions
    pred = model.predict(X_test)

    # If output path does not exist, create it
    out_path = os.path.dirname(out_file)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    savemat(out_file, mdict={'out_pred': pred})

