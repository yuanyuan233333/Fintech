from sklearn.ensemble import AdaBoostRegressor
import importlib


def get_model(parameters, state, reward, action):
    inner_model_name = parameters['adaboost']['inner_model']
    model_module = importlib.import_module('.' + inner_model_name, 'src.models')
    inner_parameters = {key: parameters[key] for key in parameters if key not in ['adaboost']}
    inner_model = model_module.get_model(inner_parameters,
                                         state.copy(),
                                         reward,
                                         action)

    adaboost_parameters = {key: parameters['adaboost'][key] for key in parameters['adaboost'] if
                           key not in ['inner_model']}
    model = AdaBoostRegressor(base_estimator=inner_model, **adaboost_parameters)
    state['action'] = action
    model.fit(state, reward)
    return model


def name():
    return 'I am the AdaBoostRegressor model'
