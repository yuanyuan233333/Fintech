from sklearn import linear_model


def get_model(parameters, state, reward, action):
    if parameters['lambda'] == 0:
        model = linear_model.LinearRegression()
    else:
        model = linear_model.Lasso(alpha=parameters['lambda'])
    state['action'] = action
    model.fit(state, reward)

    return model


def name():
    return 'I am the Lasso model'
