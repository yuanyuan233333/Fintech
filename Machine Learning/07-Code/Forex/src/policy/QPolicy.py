import numpy as np
import pandas as pd

from src.utils import check_keras_interface
from src.rewards.Position import Position


class QPolicy(object):

    def __init__(self, Q):
        check_keras_interface(Q)
        self.Q = Q

    def apply(self, minutes, state, possible_actions):
        self.check_state(state)
        self.check_indices(minutes, state)
        the_state = state.copy(deep=True)
        action = pd.Series(index=the_state.index)
        first_minute = self.build_first_minute_series(minutes)
        for minute in np.sort(minutes.unique()):
            indices, numeric_indices = self.get_indices_for(minute, minutes)
            first_of_day = first_minute[indices]
            the_state.loc[indices[first_of_day], 'position'] = Position.F
            not_first_of_day = np.logical_not(first_of_day)
            if np.any(not_first_of_day):
                the_state.loc[indices[not_first_of_day], 'position'] = action.to_numpy()[
                    numeric_indices[not_first_of_day] - 1]
            action[indices] = self.calculate_optimal_action(the_state.loc[indices, :], possible_actions)
        return the_state, action

    @staticmethod
    def build_first_minute_series(minutes):
        first_minute = minutes.values[1:] < minutes.values[:-1]
        return pd.Series(data=np.concatenate(([True], first_minute)),
                         index=minutes.index)

    @staticmethod
    def get_indices_for(minute, minutes):
        indices = minutes.index[minutes == minute]
        numeric_indices = np.arange(len(minutes.index))[minutes == minute]
        return indices, numeric_indices

    def calculate_optimal_action(self, state, possible_actions):
        Q_values = np.empty([state.shape[0], len(possible_actions)])
        for index, action in enumerate(possible_actions):
            input_for_model = state.copy(deep=True)
            input_for_model['action'] = action
            Q_values[:, index] = np.ravel(self.Q.predict(input_for_model))
        return np.array(possible_actions)[np.argmax(Q_values, axis=1)]

    @staticmethod
    def check_state(state):
        if 'position' not in state.columns:
            raise ValueError('state must be a pandas data frame with a column position')

    @staticmethod
    def check_indices(minutes, state):
        if not minutes.index.equals(state.index):
            raise ValueError('minutes must have the same index as state')
