from src.fqi.FQI import FQI
from src.rewards.Position import Position
from unittest.mock import Mock
import pytest
import numpy as np
import pandas as pd

from src.rewards.UnrealizedReward import UnrealizedReward


def fake_logger():
    return Mock(spec=['info'])


def test_raises_if_sample_doesnt_have_expected_attributes():
    with pytest.raises(ValueError):
        FQI({'current_state': [], 'next_state': [], 'reward': []}, None, {'possible_actions': []}, fake_logger())

    with pytest.raises(ValueError):
        FQI({'current_state': [], 'next_state': [], 'action': []}, None, {'possible_actions': []}, fake_logger())

    with pytest.raises(ValueError):
        FQI({'current_state': [], 'reward': [], 'action': []}, None, {'possible_actions': []}, fake_logger())

    with pytest.raises(ValueError):
        FQI({'next_state': [], 'reward': [], 'action': []}, None, {'possible_actions': []}, fake_logger())


def test_raises_if_samples_dimensions_not_congruent():
    with pytest.raises(ValueError):
        FQI({'current_state': np.array([0]), 'next_state': np.array([]), 'reward': [], 'action': []},
            None, {'possible_actions': []}, fake_logger())

    with pytest.raises(ValueError):
        FQI({'current_state': np.array([0]), 'next_state': np.array([0]), 'reward': np.array([]), 'action': []},
            None, {'possible_actions': []}, fake_logger())

    with pytest.raises(ValueError):
        FQI({'current_state': np.array([0]), 'next_state': np.array([0]), 'reward': np.array([0]),
             'action': np.array([])}, None, {'possible_actions': []}, fake_logger())


def test_raises_if_model_does_not_have_necessary_methods():
    samples = {'current_state': np.array([0]), 'next_state': np.array([0]), 'reward': np.array([0]),
               'action': np.array([0])}

    with pytest.raises(ValueError):
        model = Mock(spec=['fit'])
        FQI(samples, model, {'possible_actions': []}, fake_logger())

    with pytest.raises(ValueError):
        model = Mock(spec=['predict'])
        FQI(samples, model, {'possible_actions': []}, fake_logger())


def test_configuration_must_have_field_possible_actions():
    samples = {'current_state': np.array([0]), 'next_state': np.array([0]), 'reward': np.array([0]),
               'action': np.array([0])}
    model = Mock(spec=['fit', 'predict'])
    with pytest.raises(ValueError):
        FQI(samples, model, {}, fake_logger())


def test_configuration_uses_default_values_if_not_passed():
    samples = {'current_state': np.array([0]), 'next_state': np.array([0]), 'reward': np.array([0]),
               'action': np.array([0])}
    model = Mock(spec=['fit', 'predict'])
    fqi = FQI(samples, model, {'possible_actions': []}, fake_logger())
    assert fqi.configuration['max_iterations'] == 5
    assert fqi.configuration['discount'] == 0.99
    assert fqi.configuration['sample_iterations'] == 1


def test_configuration_overrides_defaults():
    samples = {'current_state': np.array([0]), 'next_state': np.array([0]), 'reward': np.array([0]),
               'action': np.array([0])}
    model = Mock(spec=['fit', 'predict'])
    fqi = FQI(samples, model, {'possible_actions': [], 'max_iterations': 10, 'discount': 0.9}, fake_logger())
    assert fqi.configuration['max_iterations'] == 10
    assert fqi.configuration['discount'] == 0.9


def predict_returns_100_for_10L_and_200_for_20L_else_zero(input_of_predict):
    next_state = input_of_predict.iloc[:, 0]
    action = input_of_predict.iloc[:, 3]
    max_value_for_Q = 100 * (next_state == 10) + 200 * (next_state == 20)
    return max_value_for_Q * (action == 0)


def test_build_training_set():
    samples = {'current_state': pd.DataFrame(np.array([[3, 0], [4, -1]]), columns=['a','position']),
               'next_state': pd.DataFrame(np.array([[10, 0], [20, 0]]), columns=['a','position']),
               'reward': pd.Series(np.array([1, 2])),
               'action': pd.Series(np.array([Position.L, Position.F]))}
    model = Mock(spec=['fit', 'predict'])
    model.predict.side_effect = predict_returns_100_for_10L_and_200_for_20L_else_zero
    discount = 0.9
    fqi = FQI(samples, model, {'possible_actions': [Position.L, Position.F, Position.S],
                               'discount': discount, 'max_iterations': 1}, fake_logger())
    input_value, output_value = fqi.build_training_set(samples)

    assert output_value[0] == samples['reward'][0] + discount * 100
    assert output_value[1] == samples['reward'][1] + discount * 200

    assert input_value.iloc[0, 0] == samples['current_state'].iloc[0, 0]
    assert input_value.iloc[0, 2] == samples['action'][0]
    assert input_value.iloc[1, 0] == samples['current_state'].iloc[1, 0]
    assert input_value.iloc[1, 2] == samples['action'][1]


def test_main_cycle():
    samples = {'current_state': pd.DataFrame(np.array([[3], [4]])),
               'next_state': pd.DataFrame(np.array([[10], [20]])),
               'reward': pd.Series(np.array([1, 2])),
               'action': pd.Series(np.array([Position.S, Position.F]))}
    model = Mock(spec=['fit', 'predict'])
    model.predict.side_effect = (lambda x: 100)
    discount = 0.9
    fqi = FQI(samples, model, {'possible_actions': [Position.L, Position.F, Position.S],
                               'discount': discount, 'max_iterations': 1}, fake_logger())

    fqi.run()
    model.fit.assert_called_once()


def test_run_uses_kwargs_correctly():
    samples = {'current_state': pd.DataFrame(np.array([[3]])),
               'next_state': pd.DataFrame(np.array([[10]])),
               'reward': pd.Series(np.array([1])),
               'action': pd.Series(np.array([Position.S]))}
    model = Mock(spec=['fit', 'predict'])
    model.predict.side_effect = (lambda x: 100)
    discount = 0.9
    fqi = FQI(samples, model, {'possible_actions': [Position.L, Position.F, Position.S],
                               'discount': discount, 'max_iterations': 1}, fake_logger())
    optimizer_arguments = {'epoch': 10, 'batch_size': 1000}
    fqi.run(**optimizer_arguments)
    _, kwargs = model.fit.call_args
    assert kwargs['epoch'] == optimizer_arguments['epoch']
    assert kwargs['batch_size'] == optimizer_arguments['batch_size']


def predict_return_highest_for(position):
    def predict(input_for_model):
        action = input_for_model.iloc[:, -1]
        return 100 * (action == position)

    return predict


def test_build_samples():
    samples = {'current_state': pd.DataFrame({'open': [3, 4], 'position': [Position.L, Position.S]}),
               'next_state': pd.DataFrame({'open': [10, 20], 'position': [Position.S, Position.F]}),
               'reward': pd.Series(np.array([1, 2])),
               'action': pd.Series(np.array([Position.S, Position.F])),
               'minute': pd.Series([1, 2]),
               'fee': 0.01,
               'position_size': 1000,
               'price_info': pd.DataFrame({'open': [30, 40], 'close': [50, 60]})}
    model = Mock(spec=['fit', 'predict'])
    model.predict.side_effect = predict_return_highest_for(Position.L)
    discount = 0.9
    fqi = FQI(samples, model, {'possible_actions': [Position.L, Position.F, Position.S],
                               'discount': discount, 'max_iterations': 1, 'sample_iterations': 2}, fake_logger())

    built_samples = fqi.build_samples()

    reward = UnrealizedReward(samples['fee'], samples['position_size']) \
        .calculate(built_samples['current_state'], built_samples['action'], samples['price_info'], samples['minute'])

    assert np.array(built_samples['current_state'] == np.array([[3, Position.F], [4, Position.L]])).all()
    assert np.array(built_samples['next_state'] == np.array([[10, Position.L], [20, Position.L]])).all()
    assert np.array(built_samples['action'] == np.array([Position.L, Position.L])).all()
    assert np.array(built_samples['reward'] == reward).all()


def test_2_sample_iterations():
    samples = {'current_state': pd.DataFrame({'open': [3, 4], 'position': [Position.L, Position.S]}),
               'next_state': pd.DataFrame({'open': [10, 20], 'position': [Position.S, Position.F]}),
               'reward': pd.Series(np.array([1, 2])),
               'action': pd.Series(np.array([Position.S, Position.F])),
               'minute': pd.Series([1, 2]),
               'fee': 0.01,
               'position_size': 1000,

               'price_info': pd.DataFrame({'open': [30, 40], 'close': [50, 60]})}

    model = Mock(spec=['fit', 'predict'])
    model.predict.side_effect = predict_return_highest_for(Position.L)
    discount = 0.9
    fqi = FQI(samples, model, {'possible_actions': [Position.L, Position.F, Position.S],
                               'discount': discount, 'max_iterations': 1, 'sample_iterations': 2}, fake_logger())

    fqi.run()
    model.fit.assert_called()


call_counter = 0


def predict(input_for_predict):
    global call_counter
    call_counter = call_counter + 1
    return np.full((input_for_predict.shape[0],), 100 * call_counter, dtype=int)


def test_run_returns_q_evaluations():
    samples = {'current_state': pd.DataFrame(np.array([[3], [4]])),
               'next_state': pd.DataFrame(np.array([[10], [20]])),
               'reward': pd.Series(np.array([1, 2])),
               'action': pd.Series(np.array([Position.S, Position.F]))}
    model = Mock(spec=['fit', 'predict'])

    model.predict.side_effect = predict
    discount = 0.9
    fqi = FQI(samples, model, {'possible_actions': [Position.L, Position.F, Position.S],
                               'discount': discount, 'max_iterations': 2}, fake_logger())

    _, q_evaluations, _ = fqi.run()

    assert len(q_evaluations) == 1
    assert len(q_evaluations[0]) == 1
    assert q_evaluations[0][0] == np.linalg.norm(np.array([[700, 700, 700], [700, 700, 700]]))/6


def test_run_returns_q_evaluations_2_sample_iterations():
    samples = {'current_state': pd.DataFrame({'open': [3, 4], 'position': [Position.L, Position.S]}),
               'next_state': pd.DataFrame({'open': [10, 20], 'position': [Position.S, Position.F]}),
               'reward': pd.Series(np.array([1, 2])),
               'action': pd.Series(np.array([Position.S, Position.F])),
               'minute': pd.Series([1, 2]),
               'fee': 0.01,
               'position_size': 1000,

               'price_info': pd.DataFrame({'open': [30, 40], 'close': [50, 60]})}

    model = Mock(spec=['fit', 'predict'])
    model.predict.side_effect = (lambda x: np.full((x.shape[0],), 200, dtype=int))
    discount = 0.9
    fqi = FQI(samples, model, {'possible_actions': [Position.L, Position.F, Position.S],
                               'discount': discount, 'max_iterations': 2, 'sample_iterations': 2}, fake_logger())

    _, q_evaluations, _ = fqi.run()
    assert len(q_evaluations) == 2
    assert len(q_evaluations[0]) == 1
    assert q_evaluations[0][0] == 0

    assert len(q_evaluations[1]) == 1
    assert q_evaluations[1][0] == 0
