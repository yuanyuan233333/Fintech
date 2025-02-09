from sklearn.ensemble import RandomForestRegressor


def get_model(parameters, state, reward, action):
    model = RandomForestRegressor(**parameters)
    state['action'] = action
    model.fit(state, reward)
    return model


def name():
    return 'I am the Random Forest model'
