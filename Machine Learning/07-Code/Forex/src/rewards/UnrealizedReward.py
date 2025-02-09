from src.rewards.Reward import Reward
import numpy as np
import pandas as pd


class UnrealizedReward(Reward):

    def __init__(self, fee, position_size):
        super().__init__(fee=fee, position_size=position_size)

    def calculate(self, state, action_series, price_info, minutes):
        super().check_state(state, price_info)
        action = action_series.astype('float')
        action = self.force_flat_if_last_minute(action, minutes)
        fees = self.position_size * self.fee * np.abs(state['position'].astype('int') - action)
        return pd.Series(np.array(self.position_size * state['position'].astype('int')) * np.array(
            (price_info['close'] - price_info['open'])) / np.array(price_info['close']) - fees)
