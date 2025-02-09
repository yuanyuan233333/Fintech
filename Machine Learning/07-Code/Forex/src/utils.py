import numpy as np
import pandas as pd
from src.rewards.Position import Position
from sklearn.tree import export_graphviz
from graphviz import Source
from src.rewards.UnrealizedReward import UnrealizedReward
from sklearn.preprocessing import OneHotEncoder


def check_keras_interface(model):
    predict = getattr(model, "predict", None)
    if not callable(predict):
        raise ValueError('model is missing method predict')
    fit = getattr(model, "fit", None)
    if not callable(fit):
        raise ValueError('model is missing method fit')


def to_2d_array(vector):
    return vector.reshape(vector.shape[0], 1)


def set_position(input_set, output_set, action, minutes):
    output_set['position'] = action
    input_set['position'] = np.insert(np.array(action)[:-1], 0, Position.F)
    output_set['position'] = pd.Categorical(output_set['position'])
    input_set['position'] = pd.Categorical(input_set['position'])
    first_minute_index = \
        np.hstack((input_set.index[0], np.ravel(np.where(np.array(minutes[1:]) < np.array(minutes[0:-1])))))

    input_set.loc[first_minute_index, 'position'] = Position.F
    return input_set, output_set


def plot_tree(dataset_for_fit, model):
    estimator = model.estimators_[0]
    export_graphviz(estimator, out_file='tree.dot',
                    feature_names=dataset_for_fit.columns,
                    rounded=True, proportion=False,
                    precision=2, filled=True)

    s = Source.from_file('tree.dot')
    s.view()


def create_all_combination(samples, possible_actions):
    df_void = pd.DataFrame()
    s_void = pd.Series()

    concatenated_sample = {'current_state': df_void, 'next_state': df_void, 'reward': s_void,
                                    'action': df_void}
    for action in possible_actions:
        for position in possible_actions:
            current_state = samples['current_state'].copy(deep=True)
            current_state['position'] = position
            next_state = samples['next_state'].copy(deep=True)
            next_state.loc[:, 'position'] = action
            reward = UnrealizedReward(samples['fee'], samples['position_size']) \
                    .calculate(current_state, pd.Series(np.zeros(current_state.shape[0]) * action),
                               samples['price_info'], samples['minute'])
            next_sample = {'current_state': current_state, 'next_state': next_state, 'reward': reward,
                               'action': pd.Series(np.ones(current_state.shape[0]) * action)}
            concatenated_sample = concatenate_samples(concatenated_sample, next_sample)
    return {**samples, **concatenated_sample}


def concatenate_samples(samples, new_samples):
    current_state = pd.concat([samples['current_state'], new_samples['current_state']])
    next_state = pd.concat([samples['next_state'], new_samples['next_state']])
    reward = pd.concat([samples['reward'], new_samples['reward']])
    action = pd.concat([samples['action'], new_samples['action']])
    return {'current_state': current_state, 'next_state': next_state, 'reward': reward, 'action': action}


def is_parallelizable(model):
    return 'Regressor' in str(type(model)) and 'AdaBoost' not in str(type(model)) and 'MLPRegressor' not in str(type(model))


def encoding(df, Possible_Actions):
    key = pd.DataFrame({'position': Possible_Actions, 'action': Possible_Actions})
    enc = OneHotEncoder(categories='auto')
    temp = np.array(df.loc[:, ['position', 'action']])
    enc.fit(key)
    df = df.drop(['position', 'action'], axis=1)
    return pd.concat([df, pd.DataFrame(enc.transform(temp).toarray())], axis=1)


def de_encoding(df, Possible_Actions):
    key = pd.DataFrame({'position': Possible_Actions, 'actions': Possible_Actions})
    enc = OneHotEncoder(categories='auto')
    enc.fit(key)
    columns = [i for i in range(0, (len(Possible_Actions) * 2))]
    temp = np.array(df.loc[:, columns])

    df = df.drop(columns, axis=1)
    return pd.concat([df, pd.DataFrame(enc.inverse_transform(temp), columns=['position', 'action'])], axis=1)
