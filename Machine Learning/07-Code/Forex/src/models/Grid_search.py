from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


def get_model(parameters, state, reward, action):
    base_model_parameters = {key: parameters[key] for key in parameters if key not in ['grid']}
    base_model = RandomForestRegressor(**base_model_parameters)
    model = GridSearchCV(base_model, parameters['grid']['param_grid'], **parameters['grid']['other'])
    state['action'] = action
    model.fit(state, reward)
    return model


def name():
    return 'I am the Grid_search model'
