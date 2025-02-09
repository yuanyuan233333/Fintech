from tensorflow.keras import Sequential
from tensorflow.keras.initializers import Zeros
from tensorflow.keras.layers import Dense, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam


def get_model(parameters, state, reward, action):
    initializer = Zeros()
    model = Sequential()
    for index, layer in enumerate(parameters['layers']):
        if index == 0:
            model.add(
                Dense(layer['nodes'], use_bias=True, kernel_initializer=initializer, bias_initializer=initializer,
                      input_shape=(state.shape[1] + 1,)))
        else:
            model.add(
                Dense(layer['nodes'], use_bias=True, kernel_initializer=initializer, bias_initializer=initializer))
        model.add(BatchNormalization())
        model.add(Activation(layer['activation']))
    model.add(Dense(1, activation='linear'))

    optimizer = Adam(learning_rate=parameters['optimizer']['learning_rate'],
                     beta_1=parameters['optimizer']['beta_1'],
                     beta_2=parameters['optimizer']['beta_2'])

    model.compile(optimizer=optimizer,
                  loss=parameters['loss'],
                  metrics=parameters['metrics'])

    return model


def name():
    return 'I am the Neural Network model'
