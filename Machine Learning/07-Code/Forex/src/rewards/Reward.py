import numpy as np


class Reward(object):

    def __init__(self, fee, position_size):
        if not 0 <= fee <= 1:
            raise ValueError('fee must be a percentage')
        if position_size <= 0:
            raise ValueError('position_size must be positive')
        self.fee = fee
        self.position_size = position_size

    @staticmethod
    def check_state(state, price_info):
        if 'position' not in state.columns:
            raise ValueError('column position not present')
        if 'close' not in price_info.columns:
            raise ValueError('column close not present')
        if 'open' not in price_info.columns:
            raise ValueError('column open not present')

    @staticmethod
    def force_flat_if_last_minute(action, minutes):
        last_of_day = np.concatenate((minutes.to_numpy()[:-1] > minutes.to_numpy()[1:], [True]))
        return np.logical_not(last_of_day) * action.to_numpy()
