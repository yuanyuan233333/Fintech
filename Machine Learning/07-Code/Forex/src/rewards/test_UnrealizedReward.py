from src.rewards.UnrealizedReward import UnrealizedReward
from src.rewards.Position import Position
import pytest
import pandas as pd


def build_state(minute, position):
    return pd.DataFrame({
        'minute': [minute],
        'position': [position]
    })


def build_price_info(open_price, close_price):
    return pd.DataFrame({
        'open': [open_price],
        'close': [close_price]
    })


def test_fee_must_be_percentage():
    with pytest.raises(ValueError):
        UnrealizedReward(2, None)


def test_position_size_must_be_positive():
    with pytest.raises(ValueError):
        UnrealizedReward(0, -10)


def test_state_must_have_necessary_fields():
    with pytest.raises(ValueError):
        state = pd.DataFrame({'wrong_column': [0]})
        price_info = pd.DataFrame({'wrong_column': [0]})
        minutes = pd.Series([0])
        action = pd.Series([Position.L])
        UnrealizedReward(0, 100).calculate(state, action, price_info, minutes)


def test_correct_state_does_not_raise():
    state = build_state(0, Position.F)
    price_info = build_price_info(1, 1)
    action = pd.Series([Position.F])
    minutes = pd.Series([0])

    UnrealizedReward(0, 100).calculate(state, action, price_info, minutes)


def test_from_long_position():
    # position_size in EUR, prices in EUR_USD, fees in percentage
    position_size = 100
    open_price = 1.1
    close_price = 1.2
    fees = 0.01

    state = pd.DataFrame({

        'position': [Position.L, Position.L]
    })

    price_info = pd.DataFrame({
        'open': [open_price, open_price],
        'close': [close_price, close_price]
    })

    minutes = pd.Series([0, 1])

    action = pd.Series([Position.L, Position.F])
    reward = UnrealizedReward(fees, position_size).calculate(state, action, price_info, minutes)
    assert reward[0] == position_size * (close_price - open_price) / close_price

    action = pd.Series([Position.F, Position.F])
    reward = UnrealizedReward(fees, position_size).calculate(state, action, price_info, minutes)
    assert reward[0] == position_size * (close_price - open_price) / close_price - fees * position_size

    action = pd.Series([Position.S, Position.F])
    reward = UnrealizedReward(fees, position_size).calculate(state, action, price_info, minutes)
    assert reward[0] == position_size * (close_price - open_price) / close_price - 2 * fees * position_size


def test_from_short_position():
    # position_size in EUR, prices in EUR_USD, fees in percentage
    position_size = 100
    open_price = 1.2
    close_price = 1.1
    fees = 0.01
    minutes = pd.Series([0, 1])

    state = pd.DataFrame({
        'position': [Position.S, Position.S],
    })
    price_info = pd.DataFrame({
        'open': [open_price, open_price],
        'close': [close_price, close_price]})

    action = pd.Series([Position.S, Position.F])
    reward = UnrealizedReward(fees, position_size).calculate(state, action, price_info, minutes)
    assert reward[0] == -position_size * (close_price - open_price) / close_price

    action = pd.Series([Position.F, Position.F])
    reward = UnrealizedReward(fees, position_size).calculate(state, action, price_info, minutes)
    assert reward[0] == -position_size * (close_price - open_price) / close_price - fees * position_size

    action = pd.Series([Position.L, Position.F])
    reward = UnrealizedReward(fees, position_size).calculate(state, action, price_info, minutes)
    assert reward[0] == -position_size * (close_price - open_price) / close_price - 2 * fees * position_size


def test_from_flat_position():
    # position_size in EUR, prices in EUR_USD, fees in percentage
    position_size = 100
    open_price = 1.2
    close_price = 1.1
    fees = 0.01
    minutes = pd.Series([0, 1])

    state = pd.DataFrame({
        'position': [Position.F, Position.F],
    })
    price_info = pd.DataFrame({
        'open': [open_price, open_price],
        'close': [close_price, close_price]})
    action = pd.Series([Position.F, Position.F])
    reward = UnrealizedReward(fees, position_size).calculate(state, action, price_info, minutes)
    assert reward[0] == 0

    action = pd.Series([Position.L, Position.F])
    reward = UnrealizedReward(fees, position_size).calculate(state, action, price_info, minutes)
    assert reward[0] == - fees * position_size

    action = pd.Series([Position.S, Position.F])
    reward = UnrealizedReward(fees, position_size).calculate(state, action, price_info, minutes)
    assert reward[0] == - fees * position_size


@pytest.mark.parametrize("action", [Position.L, Position.F, Position.S])
def test_end_of_day_is_forced_flat(action):
    position_size = 100
    open_price = 1.1
    close_price = 1.2
    fees = 0.01
    minutes = pd.Series([100, 1])

    # from long
    state = pd.DataFrame({
        'position': [Position.L, Position.L],
    })
    price_info = pd.DataFrame({
        'open': [open_price, open_price],
        'close': [close_price, close_price]})

    reward = UnrealizedReward(fees, position_size).calculate(state, pd.Series([action, action]), price_info, minutes)
    assert reward[0] == position_size * (close_price - open_price) / close_price - fees * position_size

    # from short
    state = pd.DataFrame({
        'minute': [100, 1],
        'position': [Position.S, Position.S],

    })
    price_info = pd.DataFrame({
        'open': [open_price, open_price],
        'close': [close_price, close_price]})
    reward = UnrealizedReward(fees, position_size).calculate(state, pd.Series([action, action]), price_info, minutes)
    assert reward[0] == -position_size * (close_price - open_price) / close_price - fees * position_size

    # from flat
    state = pd.DataFrame({
        'minute': [100, 1],
        'position': [Position.F, Position.F],
    })
    reward = UnrealizedReward(fees, position_size).calculate(state, pd.Series([action, action]), price_info, minutes)
    assert reward[0] == 0
