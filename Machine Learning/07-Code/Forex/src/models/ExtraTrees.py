from sklearn.ensemble import ExtraTreesRegressor


def get_model(parameters, state, reward, action):
    model = ExtraTreesRegressor(**parameters)
    state['action'] = action
    model.fit(state, reward)
    return model


def name():
    return 'I am the Extremely Randomized Trees model'
