import numpy as np

from src.policy.QPolicy import QPolicy
from src.rewards.Position import Position
from src.rewards.UnrealizedReward import UnrealizedReward
from src.utils import check_keras_interface, concatenate_samples, is_parallelizable

class FQI:
    default_configuration = {'max_iterations': 5, 'discount': 0.99, 'sample_iterations': 1}

    def __init__(self, samples, model, configuration, logger):
        self.check_samples(samples)
        check_keras_interface(model)
        self.check_configuration(configuration)
        self.samples = samples
        self.model = model
        self.configuration = {**FQI.default_configuration, **configuration}
        self.logger = logger

    def run(self, **kwargs):
        q_sample_norms = []
        fitting_sample_losses = []
        for sample_iteration in range(self.configuration['sample_iterations']):
            samples = self.samples if sample_iteration == 0 else concatenate_samples(samples, self.build_samples())
            q_norms = []
            fitting_losses = []
            previousQ = None
            for iteration in range(self.configuration['max_iterations']):
                self.logger.info("Q-iteration {}".format(iteration))
                input_values, output_values = self.build_training_set(samples)
                self.model.fit(input_values, output_values, **kwargs)
                Q = self.evaluate_Q(samples['current_state'])
                q_norms.append(np.linalg.norm(Q - previousQ)/Q.size if previousQ is not None else 0)
                previousQ = Q
                fitting_losses.append(self.calculate_loss(input_values, output_values))
            q_sample_norms.append(q_norms[1:])
            fitting_sample_losses.append(fitting_losses)
        return self.model, q_sample_norms, fitting_sample_losses

    def build_samples(self):
        if is_parallelizable(self.model):
            self.model.set_params(n_jobs=1)
        policy = QPolicy(self.model)
        current_state, action = policy.apply(self.samples['minute'],
                                             self.samples['current_state'],
                                             self.configuration['possible_actions'])
        next_state = self.samples['next_state'].copy(deep=True)
        next_state.loc[:, 'position'] = action
        reward = UnrealizedReward(self.samples['fee'], self.samples['position_size']) \
            .calculate(current_state, action, self.samples['price_info'], self.samples['minute'])
        if is_parallelizable(self.model):
            self.model.set_params(n_jobs=-1)
        return {'current_state': current_state, 'next_state': next_state, 'reward': reward, 'action': action}

    def build_training_set(self, samples):
        input_values = samples['current_state']
        input_values['action'] = samples['action']
        output_values = np.ravel(samples['reward']) + self.configuration['discount'] * self.max_Q(samples['next_state'])
        return input_values, output_values

    def max_Q(self, state):
        Q = self.evaluate_Q(state)
        return np.amax(Q, axis=1)

    def evaluate_Q(self, state):
        possible_actions = self.configuration['possible_actions']
        Q = np.empty([state.shape[0], len(possible_actions)])
        for index, action in enumerate(possible_actions):
            input_for_model = state.copy()
            input_for_model['action'] = action
            Q[:, index] = np.ravel(self.model.predict(input_for_model))
        return Q

    @staticmethod
    def check_samples(samples):
        FQI.check_fields(samples)
        FQI.check_dimensions(samples)

    @staticmethod
    def check_fields(samples):
        if 'action' not in samples:
            raise ValueError('missing necessary field action')
        if 'reward' not in samples:
            raise ValueError('missing necessary field reward')
        if 'next_state' not in samples:
            raise ValueError('missing necessary field next_state')
        if 'current_state' not in samples:
            raise ValueError('missing necessary field current_state')

    @staticmethod
    def check_dimensions(samples):
        if samples['current_state'].shape != samples['next_state'].shape:
            raise ValueError('current_state and next_state do not have same dimensions')
        if samples['current_state'].shape[0] != samples['reward'].shape[0]:
            raise ValueError('current_state and reward do not have same number of observations')
        if samples['current_state'].shape[0] != samples['action'].shape[0]:
            raise ValueError('current_state and action do not have same number of observations')

    @staticmethod
    def check_configuration(configuration):
        if 'possible_actions' not in configuration:
            raise ValueError('missing necessary field possible_actions in configuration parameter')

    def calculate_loss(self, input_values, output_values):
        return np.sum((np.ravel(self.model.predict(input_values)) - np.ravel(output_values)) ** 2) / output_values.size

