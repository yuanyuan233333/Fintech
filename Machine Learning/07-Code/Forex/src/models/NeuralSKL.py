from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def get_model(parameters, state, reward, action):
    model = MLPRegressor(**parameters)
    state['action'] = action
    pipe_model = Pipeline([
        ('scale', StandardScaler()),
        ('MLPRegressor', model)
    ])
    pipe_model.fit(state, reward)
    return pipe_model


def name():
    return 'I am the Neural Net model by SKLearn'
