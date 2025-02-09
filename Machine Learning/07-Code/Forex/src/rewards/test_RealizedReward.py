from src.rewards.RealizedReward import RealizedReward
from src.rewards.Position import Position
import pytest
import pandas as pd


def build_state(minute, position, open_price, close_price, position_price):
    return pd.DataFrame({
        'minute': [minute],
        'position': [position],
    })

def build_price_info(open_price, close_price):
    return pd.DataFrame({
        'open': [open_price],
        'close': [close_price],
        'position_price': [position_price]

    })

def test_fee_must_be_percentage():
    with pytest.raises(ValueError):
        RealizedReward(2, None)


def test_position_size_must_be_positive():
    with pytest.raises(ValueError):
        RealizedReward(0, -10)


def test_state_must_have_necessary_fields():
    with pytest.raises(ValueError):
        state = pd.DataFrame({'wrong_column': [0]})
        price_info = pd.DataFrame({'wrong_column': [0]})
        action = pd.Series([Position.L])
        minutes=pd.Series([0])
        RealizedReward(0, 100).calculate(state, action, price_info, minutes)


def test_raise_if_missing_position_price():
    with pytest.raises(ValueError):
        minutes=pd.Series([0])

        state = pd.DataFrame({
            'minute': [0],
            'position': [Position.F]

        })
        price_info=pd.DataFrame({
            'open': [0],
            'close': [0]
        })
        action = pd.Series([Position.F])
        RealizedReward(0, 100).calculate(state, action,price_info, minutes)


def test_from_long_position():
    # position_size in EUR, prices in EUR_USD, fees in percentage
    position_size = 100
    open_price = 1.1
    close_price = 1.2
    position_price = 1.15
    fees = 0.01
    minutes = pd.Series([0, 1])

    state = pd.DataFrame({
        'minute': [0, 1],
        'position': [Position.L, Position.L],
    })
    price_info=pd.DataFrame({
        'open': [open_price, open_price],
        'close': [close_price, close_price],
        'position_price': [position_price, position_price],

    })


    action = pd.Series([Position.L, Position.F])
    reward = RealizedReward(fees, position_size).calculate(state, action, price_info, minutes)
    assert reward[0] == 0

    action = pd.Series([Position.F, Position.F])
    reward = RealizedReward(fees, position_size).calculate(state, action, price_info, minutes)
    assert reward[0] == position_size * (close_price - position_price) / close_price - fees * position_size

    action = pd.Series([Position.S, Position.F])
    reward = RealizedReward(fees, position_size).calculate(state, action, price_info, minutes)
    assert reward[0] == position_size * (close_price - position_price) / close_price - 2 * fees * position_size


def test_from_short_position():
    # position_size in EUR, prices in EUR_USD, fees in percentage
    position_size = 100
    open_price = 1.2
    close_price = 1.1
    position_price = 1.15
    fees = 0.01
    minutes = pd.Series([0, 1])

    state = pd.DataFrame({
        'minute': [0, 1],
        'position': [Position.S, Position.S],
    })
    price_info=pd.DataFrame({
        'open': [open_price, open_price],
        'close': [close_price, close_price],
        'position_price': [position_price, position_price],

    })

    action = pd.Series([Position.S, Position.F])
    reward = RealizedReward(fees, position_size).calculate(state, action, price_info, minutes)
    assert reward[0] == 0

    action = pd.Series([Position.F, Position.F])
    reward = RealizedReward(fees, position_size).calculate(state, action, price_info, minutes)
    assert reward[0] == -position_size * (close_price - position_price) / close_price - fees * position_size

    action = pd.Series([Position.L, Position.F])
    reward = RealizedReward(fees, position_size).calculate(state, action, price_info, minutes)
    assert reward[0] == -position_size * (close_price - position_price) / close_price - 2 * fees * position_size


def test_from_flat_position():
    # position_size in EUR, prices in EUR_USD, fees in percentage
    position_size = 100
    open_price = 1.2
    close_price = 1.1
    position_price = 1.15
    fees = 0.01
    minutes = pd.Series([0, 1])

    state = pd.DataFrame({
        'minute': [0, 1],
        'position': [Position.F, Position.F]
    })
    price_info=pd.DataFrame({
        'open': [open_price, open_price],
        'close': [close_price, close_price],
        'position_price': [position_price, position_price],

    })
    action = pd.Series([Position.F, Position.F])
    reward = RealizedReward(fees, position_size).calculate(state, action, price_info, minutes)
    assert reward[0] == 0

    action = pd.Series([Position.L, Position.F])
    reward = RealizedReward(fees, position_size).calculate(state, action, price_info,  minutes)
    assert reward[0] == - fees * position_size

    action = pd.Series([Position.S, Position.F])
    reward = RealizedReward(fees, position_size).calculate(state, action, price_info, minutes)
    assert reward[0] == - fees * position_size


@pytest.mark.parametrize("action", [Position.L, Position.F, Position.S])
def test_end_of_day_is_forced_flat(action):
    position_size = 100
    open_price = 1.1
    close_price = 1.2
    position_price = 1.15
    fees = 0.01
    minutes = pd.Series([100, 1])

    # from long
    state = pd.DataFrame({
        'minute': [100, 1],
        'position': [Position.L, Position.L]
    })
    price_info = pd.DataFrame({
        'open': [open_price, open_price],
        'close': [close_price, close_price],
        'position_price': [position_price, position_price],

    })
    reward = RealizedReward(fees, position_size).calculate(state, pd.Series([action, action]), price_info, minutes)
    assert reward[0] == position_size * (close_price - position_price) / close_price - fees * position_size

    # from short
    state = pd.DataFrame({
        'minute': [100, 1],
        'position': [Position.S, Position.S],
    })
    price_info = pd.DataFrame({
        'open': [open_price, open_price],
        'close': [close_price, close_price],
        'position_price': [position_price, position_price],

    })
    reward = RealizedReward(fees, position_size).calculate(state, pd.Series([action, action]), price_info, minutes)
    assert reward[0] == -position_size * (close_price - position_price) / close_price - fees * position_size

    # from flat
    state = pd.DataFrame({
        'minute': [100, 1],
        'position': [Position.F, Position.F],
    })
    price_info = pd.DataFrame({
        'open': [open_price, open_price],
        'close': [close_price, close_price],
        'position_price': [position_price, position_price],

    })
    reward = RealizedReward(fees, position_size).calculate(state, pd.Series([action, action]), price_info, minutes)
    assert reward[0] == 0
