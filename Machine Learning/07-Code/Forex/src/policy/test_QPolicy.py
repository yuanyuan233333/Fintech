from src.policy.QPolicy import QPolicy
from src.rewards.Position import Position
from unittest.mock import Mock
import pytest
import pandas as pd
import datetime as dt

POSSIBLE_ACTIONS = [Position.L, Position.F, Position.S]


def test_raise_if_minutes_different_index_state():
    Q = Mock(spec=['predict', 'fit'])
    policy = QPolicy(Q)
    with pytest.raises(ValueError):
        state = pd.DataFrame(data={'position': [Position.F]}, index=['a'])
        minutes = pd.Series(data=[10], index=[1])
        policy.apply(minutes, state, POSSIBLE_ACTIONS)


def test_raises_if_Q_is_not_keras_model():
    with pytest.raises(ValueError):
        Q = Mock(spec=['predict'])
        QPolicy(Q)

    with pytest.raises(ValueError):
        Q = Mock(spec=['fit'])
        QPolicy(Q)


def test_raises_if_data_frame_is_missing_necessary_columns():
    Q = Mock(spec=['predict', 'fit'])
    policy = QPolicy(Q)
    with pytest.raises(ValueError):
        state = pd.DataFrame({'not_necessary_column': []})
        policy.apply([], state, POSSIBLE_ACTIONS)


def predict_return_highest_for(position):
    def predict(input_for_model):
        action = input_for_model.iloc[:, -1]
        return 100 * (action == position)

    return predict


def test_calculate_optimal_action():
    Q = Mock(spec=['predict', 'fit'])
    Q.predict.side_effect = predict_return_highest_for(Position.F)
    policy = QPolicy(Q)
    state = pd.DataFrame({
        'open': [1.1, 1.2],
        'close': [1.1, 1.2],
        'position': [Position.S, Position.L],
    })
    optimal_action = policy.calculate_optimal_action(state, POSSIBLE_ACTIONS)
    assert optimal_action[0] == Position.F
    assert optimal_action[1] == Position.F


def test_apply_sets_first_position_to_flat():
    Q = Mock(spec=['predict', 'fit'])
    Q.predict.side_effect = predict_return_highest_for(Position.L)
    policy = QPolicy(Q)
    state = pd.DataFrame({
        'date': ['01/04', '01/04'],
        'position': [Position.S, Position.S],
        'open': [10, 20],
        'close': [30, 40]})
    minutes = pd.Series([1, 2])
    new_state, action = policy.apply(minutes, state, POSSIBLE_ACTIONS)
    assert new_state['position'][0] == Position.F


def predict_return_100_if_10openF_or_200_if_20openL(input_for_model):
    open_price = input_for_model.iloc[:, 2]
    action = input_for_model.iloc[:, -1]
    max_value = 100 * (open_price == 10) + 200 * (open_price == 20)
    optimal_action = (open_price == 10) * (action == Position.L) + (open_price == 20) * (action == Position.F)
    return max_value * optimal_action


def test_apply_returns_correct_actions():
    Q = Mock(spec=['predict', 'fit'])
    Q.predict.side_effect = predict_return_100_if_10openF_or_200_if_20openL
    policy = QPolicy(Q)
    index = pd.Index([dt.datetime(2018, 10, 15, 18, 22), dt.datetime(2018, 10, 15, 18, 23)])
    state = pd.DataFrame(
        data={'date': ['01/04', '01/04'], 'position': [Position.S, Position.S], 'open': [10, 20],
              'close': [30, 40]},
        index=index)
    minutes = pd.Series(data=[1, 2], index=state.index)
    new_state, action = policy.apply(minutes, state, POSSIBLE_ACTIONS)
    assert action[index[0]] == Position.L
    assert action[index[1]] == Position.F


def test_apply_returns_correct_positions():
    Q = Mock(spec=['predict', 'fit'])
    Q.predict.side_effect = predict_return_100_if_10openF_or_200_if_20openL
    policy = QPolicy(Q)
    index = pd.Index([dt.datetime(2018, 10, 15, 18, 22), dt.datetime(2018, 10, 15, 18, 23)])
    state = pd.DataFrame(
        data={'date': ['01/04', '01/04'], 'position': [Position.S, Position.S], 'open': [10, 20],
              'close': [30, 40]},
        index=index)
    minutes = pd.Series(data=[1, 2], index=state.index)
    new_state, _ = policy.apply(minutes, state, POSSIBLE_ACTIONS)
    assert new_state['position'][index[0]] == Position.F
    assert new_state['position'][index[1]] == Position.L


def test_apply_does_not_modify_other_columns():
    Q = Mock(spec=['predict', 'fit'])
    Q.predict.side_effect = predict_return_100_if_10openF_or_200_if_20openL
    policy = QPolicy(Q)
    index = pd.Index([dt.datetime(2018, 10, 15, 18, 22), dt.datetime(2018, 10, 15, 18, 23)])
    state = pd.DataFrame(
        data={'date': ['01/04', '01/04'], 'minute': [1, 2], 'position': [Position.S, Position.S], 'open': [10, 20],
              'close': [30, 40]},
        index=index)
    minutes = pd.Series(data=[1, 2], index=state.index)
    new_state, _ = policy.apply(minutes, state, POSSIBLE_ACTIONS)
    assert new_state['open'][index[0]] == 10
    assert new_state['open'][index[1]] == 20
    assert new_state['close'][index[0]] == 30
    assert new_state['close'][index[1]] == 40


def test_apply_works_on_copy():
    Q = Mock(spec=['predict', 'fit'])
    Q.predict.side_effect = predict_return_100_if_10openF_or_200_if_20openL
    policy = QPolicy(Q)
    index = pd.Index([dt.datetime(2018, 10, 15, 18, 22), dt.datetime(2018, 10, 15, 18, 23)])
    state = pd.DataFrame(
        data={'date': ['01/04', '01/04'], 'position': [Position.S, Position.S], 'open': [10, 20],
              'close': [30, 40]},
        index=index)
    minutes = pd.Series(data=[1, 2], index=state.index)
    new_state, _ = policy.apply(minutes, state, POSSIBLE_ACTIONS)

    new_state.loc[index[0], 'open'] = 100
    assert state['open'][index[0]] == 10


def test_apply_sets_flat_position_for_first_minute():
    Q = Mock(spec=['predict', 'fit'])
    Q.predict.side_effect = predict_return_highest_for(Position.L)
    policy = QPolicy(Q)
    state = pd.DataFrame(
        data={'date': ['01/04', '01/04', '01/05', '01/05'],
              'position': [Position.S, Position.S, Position.S, Position.S],
              'open': [10, 20, 30, 40],
              'close': [30, 40, 50, 60]})
    minutes = pd.Series(data=[2, 3, 2, 3], index=state.index)
    new_state, _ = policy.apply(minutes, state, POSSIBLE_ACTIONS)
    assert new_state['position'][2] == Position.F


def test_apply_days_with_same_number_of_minutes():
    Q = Mock(spec=['predict', 'fit'])
    Q.predict.side_effect = predict_return_highest_for(Position.L)
    policy = QPolicy(Q)
    state = pd.DataFrame(
        data={'date': ['01/04', '01/04', '01/05', '01/05'],
              'position': [Position.S, Position.S, Position.S, Position.S],
              'open': [10, 20, 30, 40],
              'close': [30, 40, 50, 60]})
    minutes = pd.Series(data=[2, 3, 2, 3], index=state.index)
    _, action = policy.apply(minutes, state, POSSIBLE_ACTIONS)
    assert action[0] == Position.L
    assert action[1] == Position.L
    assert action[2] == Position.L
    assert action[3] == Position.L


def test_apply_days_ending_in_different_minutes():
    Q = Mock(spec=['predict', 'fit'])
    Q.predict.side_effect = predict_return_highest_for(Position.L)
    policy = QPolicy(Q)
    state = pd.DataFrame(
        data={'date': ['01/04', '01/04', '01/05', '01/05', '01/05'],
              'position': [Position.S, Position.S, Position.S, Position.S, Position.S],
              'open': [10, 20, 30, 40, 50],
              'close': [30, 40, 50, 60, 70]})
    minutes = pd.Series(data=[2, 3, 2, 3, 4], index=state.index)
    _, action = policy.apply(minutes, state, POSSIBLE_ACTIONS)
    assert action[0] == Position.L
    assert action[1] == Position.L
    assert action[2] == Position.L
    assert action[3] == Position.L
    assert action[3] == Position.L


def test_apply_days_starting_in_different_minutes():
    Q = Mock(spec=['predict', 'fit'])
    Q.predict.side_effect = predict_return_highest_for(Position.L)
    policy = QPolicy(Q)
    state = pd.DataFrame(
        data={'date': ['01/04', '01/04', '01/05', '01/05', '01/05'],
              'position': [Position.S, Position.S, Position.S, Position.S, Position.S],
              'open': [10, 20, 30, 40, 50],
              'close': [30, 40, 50, 60, 70]})
    minutes = pd.Series(data=[2, 3, 1, 2, 3], index=state.index)
    _, action = policy.apply(minutes, state, POSSIBLE_ACTIONS)
    assert action[0] == Position.L
    assert action[1] == Position.L
    assert action[2] == Position.L
    assert action[3] == Position.L
    assert action[3] == Position.L


