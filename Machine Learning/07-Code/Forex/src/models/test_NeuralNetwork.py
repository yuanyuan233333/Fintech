import pandas as pd

from src.models.NeuralNetwork import get_model


def test_layers():
    parameters = {
        'layers': [
            {'nodes': 5, 'activation': 'relu'},
            {'nodes': 2, 'activation': 'tanh'},
        ],
        'optimizer': {'learning_rate': 0.01, 'beta_1': 0.8, 'beta_2': 0.7},
        'loss': 'mean_squared_error',
        'metrics': ['accuracy']
    }
    state = pd.DataFrame({})
    model = get_model(parameters, state, pd.Series([]), pd.Series([]))
    assert len(model.layers) == 7
    assert model.layers[0].units == 5
    assert 'Relu' in str(model.layers[2].output.name)
    assert model.layers[3].units == 2
    assert 'Tanh' in str(model.layers[5].output.name)
    assert model.layers[6].units == 1
