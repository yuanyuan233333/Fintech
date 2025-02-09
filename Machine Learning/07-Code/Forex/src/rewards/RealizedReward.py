from src.rewards.Reward import Reward
import numpy as np


class RealizedReward(Reward):
    def __init__(self, fee, position_size):
        super().__init__(fee=fee, position_size=position_size)

    def check_state(self, state, price_info):
        super().check_state(state, price_info)
        if 'position_price' not in price_info.columns:
            raise ValueError('column position_price not present')

    def calculate(self, state, action, price_info,minutes):
        self.check_state(state, price_info)
        action = self.force_flat_if_last_minute(action, minutes)
        fees = self.position_size * self.fee * np.abs(state['position'] - action)
        has_changed_position = state['position'] != action
        return has_changed_position * self.position_size * state['position'] * (
                price_info['close'] - price_info['position_price']) / price_info['close'] - fees
