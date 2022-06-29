import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import deepcopy
from nested.optimize_utils import Context
from train_network_utils import *


context = Context()


class Hebb_lat_inh_network(object):

    def __init__(self, num_hidden_layers, input_dim, hidden_dim, hidden_inh_dim, output_dim, output_inh_dim, tau,
                 num_steps, activation_f=None, seed=None):
        self.num_hidden_layers = num_hidden_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_inh_dim = hidden_inh_dim
        self.output_dim = output_dim
        self.output_inh_dim = output_inh_dim
        self.tau = tau
        self.num_steps = num_steps
        self.activation_f = activation_f
        self.seed = seed
        self.random = np.random.default_rng(seed=self.seed)
        self.num_layers = num_hidden_layers + 2
        if num_hidden_layers > 0:
            if isinstance(self.hidden_dim, list):
                if len(self.hidden_dim) != self.num_hidden_layers:
                    raise Exception('Hebb_lat_inh_network: hidden_dim must be int or list of len num_hidden_layers')
                self.layer_dims = [self.input_dim] + self.hidden_dim + [self.output_dim]
                if not isinstance(self.hidden_inh_dim, list) or len(self.hidden_inh_dim) != self.num_hidden_layers:
                    raise Exception('Hebb_lat_inh_network: hidden_inh_dim must be int or list of len num_hidden_layers')
                self.inh_layer_dims = [0] + self.hidden_inh_dim + [self.output_inh_dim]

            elif isinstance(self.hidden_dim, int):
                self.layer_dims = [self.input_dim] + self.num_hidden_layers * [self.hidden_dim] + [self.output_dim]
                if not isinstance(self.hidden_inh_dim, int):
                    raise Exception('Hebb_lat_inh_network: hidden_inh_dim must be int or list of len num_hidden_layers')
                self.inh_layer_dims = [0] + self.num_hidden_layers * [self.hidden_inh_dim] + [self.output_inh_dim]
        else:
            self.layer_dims = [self.input_dim, self.output_dim]
            self.inh_layer_dims = [0, self.output_inh_dim]
    
    def init_weights(self, E_E_weight_scale_dict, E_I_weight_scale_dict, I_E_weight_scale_dict, I_I_weight_scale_dict,
                     E_E_weight_bounds_dict=None, E_I_weight_bounds_dict=None, I_E_weight_bounds_dict=None,
                     I_I_weight_bounds_dict=None):
        self.E_E_weight_scale_dict = E_E_weight_scale_dict
        self.E_I_weight_scale_dict = E_I_weight_scale_dict
        self.I_E_weight_scale_dict = I_E_weight_scale_dict
        self.I_I_weight_scale_dict = I_I_weight_scale_dict
        if E_E_weight_bounds_dict is not None:
            self.E_E_weight_bounds_dict = E_E_weight_bounds_dict
        else:
            self.E_E_weight_bounds_dict = (None, None)
        if E_I_weight_bounds_dict is not None:
            self.E_I_weight_bounds_dict = E_I_weight_bounds_dict
        else:
            self.E_I_weight_bounds_dict = (None, None)
        if I_E_weight_bounds_dict is not None:
            self.I_E_weight_bounds_dict = I_E_weight_bounds_dict
        else:
            self.I_E_weight_bounds_dict = (None, None)
        if I_I_weight_bounds_dict is not None:
            self.I_I_weight_bounds_dict = I_I_weight_bounds_dict
        else:
            self.I_I_weight_bounds_dict = (None, None)
        self.re_init_weights()

    def re_init_weights(self, seed=None):
        if seed is not None:
            self.seed = seed
        self.random = np.random.default_rng(seed=self.seed)
        self.initial_E_E_weight_matrix_dict = {}
        self.initial_E_I_weight_matrix_dict = {}
        self.initial_I_E_weight_matrix_dict = {}
        self.initial_I_I_weight_matrix_dict = {}

        for layer in range(1, self.num_layers):
            curr_layer_dim = self.layer_dims[layer]
            prev_layer_dim = self.layer_dims[layer - 1]
            self.initial_E_E_weight_matrix_dict[layer] = \
                self.random.uniform(0., 1., [curr_layer_dim, prev_layer_dim])
            curr_inh_layer_dim = self.inh_layer_dims[layer]
            if curr_inh_layer_dim > 0:
                if curr_inh_layer_dim > 1:
                    self.initial_E_I_weight_matrix_dict[layer] = \
                        self.random.uniform(-1., 0., [curr_layer_dim, curr_inh_layer_dim])
                    self.initial_I_E_weight_matrix_dict[layer] = \
                        self.random.uniform(0., 1., [curr_inh_layer_dim, curr_layer_dim])
                    self.initial_I_I_weight_matrix_dict[layer] = \
                        self.random.uniform(-1., 0., [curr_inh_layer_dim, curr_inh_layer_dim])
                else:
                    self.initial_E_I_weight_matrix_dict[layer] = -np.ones([curr_layer_dim, curr_inh_layer_dim]) * \
                                                                 self.E_I_weight_scale_dict[layer]
                    self.initial_I_E_weight_matrix_dict[layer] = np.ones([curr_inh_layer_dim, curr_layer_dim]) * \
                                                                 self.I_E_weight_scale_dict[layer]
                    self.initial_I_I_weight_matrix_dict[layer] = -np.ones([curr_inh_layer_dim, curr_inh_layer_dim]) * \
                                                                 self.I_I_weight_scale_dict[layer]

        self.initial_E_E_weight_matrix_dict, self.initial_E_I_weight_matrix_dict, \
        self.initial_I_E_weight_matrix_dict, self.initial_I_I_weight_matrix_dict = \
            self.normalize_weights(self.initial_E_E_weight_matrix_dict, self.initial_E_I_weight_matrix_dict,
                                   self.initial_I_E_weight_matrix_dict, self.initial_I_I_weight_matrix_dict)

    def normalize_weights(self, E_E_weight_matrix_dict, E_I_weight_matrix_dict, I_E_weight_matrix_dict,
                          I_I_weight_matrix_dict):
        for layer in range(1, self.num_layers):
            E_E_weight_matrix_dict[layer] = self.E_E_weight_scale_dict[layer] * E_E_weight_matrix_dict[layer] / \
                                            np.sum(np.abs(E_E_weight_matrix_dict[layer]), axis=1)[:, np.newaxis]
            curr_inh_layer_dim = self.inh_layer_dims[layer]
            if curr_inh_layer_dim > 1:
                for weight_matrix_dict, weight_scale_dict in \
                        zip([E_I_weight_matrix_dict, I_E_weight_matrix_dict, I_I_weight_matrix_dict],
                            [self.E_I_weight_scale_dict, self.I_E_weight_scale_dict, self.I_I_weight_scale_dict]):
                    weight_matrix_dict[layer] = weight_scale_dict[layer] * weight_matrix_dict[layer] / \
                                         np.sum(np.abs(weight_matrix_dict[layer]), axis=1)[:, np.newaxis]
        return E_E_weight_matrix_dict, E_I_weight_matrix_dict, I_E_weight_matrix_dict, I_I_weight_matrix_dict

    def load_input_patterns(self, input_pattern_matrix):
        if input_pattern_matrix.shape[0] != self.input_dim:
            raise Exception('Hebb_lat_inh_network.load_input_patterns: input_pattern_matrix shape does not match input_dim')
        if hasattr(self, 'target_output_pattern_matrix') and \
                self.target_output_pattern_matrix.shape[-1] != input_pattern_matrix.shape[-1]:
            raise Exception('Hebb_lat_inh_network.load_input_patterns: input_pattern_matrix shape does not match '
                            'target_output_pattern_matrix')
        self.input_pattern_matrix = input_pattern_matrix
        
    def load_target_output_patterns(self, target_output_pattern_matrix):
        if target_output_pattern_matrix.shape[0] != self.output_dim:
            raise Exception('Hebb_lat_inh_network.load_target_output_patterns: target_output_pattern_matrix shape does not '
                            'match output_dim')
        if hasattr(self, 'input_pattern_matrix') and \
                self.input_pattern_matrix.shape[-1] != target_output_pattern_matrix.shape[-1]:
            raise Exception('Hebb_lat_inh_network.load_target_output_patterns: input_pattern_matrix shape does not match '
                            'target_output_pattern_matrix')
        self.target_output_pattern_matrix = target_output_pattern_matrix

    def config_learning_rules(self, E_E_learning_rule_dict, E_I_learning_rule_dict, I_E_learning_rule_dict,
                                  I_I_learning_rule_dict, E_E_learning_rate_dict, E_I_learning_rate_dict,
                                  I_E_learning_rate_dict, I_I_learning_rate_dict):
        self.E_E_learning_rule_dict = {}
        self.E_I_learning_rule_dict = {}
        self.I_E_learning_rule_dict = {}
        self.I_I_learning_rule_dict = {}

        for layer in E_E_learning_rule_dict:
            if E_E_learning_rule_dict[layer] == 'Hebb + weight norm':
                self.E_E_learning_rule_dict[layer] = get_Hebb_rule(E_E_learning_rate_dict[layer], direction=1)
            elif E_E_learning_rule_dict[layer] == 'Anti-Hebb + weight norm':
                self.E_E_learning_rule_dict[layer] = get_anti_Hebb_rule(E_E_learning_rate_dict[layer], direction=1)
            elif E_E_learning_rule_dict[layer] is None:
                self.E_E_learning_rule_dict[layer] = null_rule
            else:
                raise Exception('Hebb_lat_inh_network.config_learning_rules: learning rule not recognized: %s' %
                                E_E_learning_rule_dict[layer])
        for layer in E_I_learning_rule_dict:
            if E_I_learning_rule_dict[layer] == 'Hebb + weight norm':
                self.E_I_learning_rule_dict[layer] = get_Hebb_rule(E_I_learning_rate_dict[layer], direction=-1)
            elif E_I_learning_rule_dict[layer] == 'Anti-Hebb + weight norm':
                self.E_I_learning_rule_dict[layer] = get_anti_Hebb_rule(E_I_learning_rate_dict[layer], direction=-1)
            elif E_I_learning_rule_dict[layer] is None:
                self.E_I_learning_rule_dict[layer] = null_rule
            else:
                raise Exception('Hebb_lat_inh_network.config_learning_rules: learning rule not recognized: %s' %
                                E_I_learning_rule_dict[layer])
        for layer in I_E_learning_rule_dict:
            if I_E_learning_rule_dict[layer] == 'Hebb + weight norm':
                self.I_E_learning_rule_dict[layer] = get_Hebb_rule(I_E_learning_rate_dict[layer], direction=1)
            elif I_E_learning_rule_dict[layer] == 'Anti-Hebb + weight norm':
                self.I_E_learning_rule_dict[layer] = get_anti_Hebb_rule(I_E_learning_rate_dict[layer], direction=1)
            elif I_E_learning_rule_dict[layer] is None:
                self.I_E_learning_rule_dict[layer] = null_rule
            else:
                raise Exception('Hebb_lat_inh_network.config_learning_rules: learning rule not recognized: %s' %
                                I_E_learning_rule_dict[layer])
        for layer in I_I_learning_rule_dict:
            if I_I_learning_rule_dict[layer] == 'Hebb + weight norm':
                self.I_I_learning_rule_dict[layer] = get_Hebb_rule(I_I_learning_rate_dict[layer], direction=-1)
            elif I_I_learning_rule_dict[layer] == 'Anti-Hebb + weight norm':
                self.I_I_learning_rule_dict[layer] = get_anti_Hebb_rule(I_I_learning_rate_dict[layer], direction=-1)
            elif I_I_learning_rule_dict[layer] is None:
                self.I_I_learning_rule_dict[layer] = null_rule
            else:
                raise Exception('Hebb_lat_inh_network.config_learning_rules: learning rule not recognized: %s' %
                                I_I_learning_rule_dict[layer])
            
    def get_layer_activities(self, input_pattern_matrix, E_E_weight_matrix_dict, E_I_weight_matrix_dict,
                             I_E_weight_matrix_dict, I_I_weight_matrix_dict):

        layer_output_dict = {}
        layer_output_dict[0] = np.copy(input_pattern_matrix)
        prev_layer_output = input_pattern_matrix
        sorted_layers = sorted(list(E_E_weight_matrix_dict.keys()))

        layer_inh_output_dict = {}
        past_layer_inh_output_dict = {}
        past_layer_state_dict = {}
        past_layer_inh_state_dict = {}

        for layer in sorted_layers:
            if len(input_pattern_matrix.shape) > 1:
                num_patterns = input_pattern_matrix.shape[-1]
                past_layer_state_dict[layer] = np.zeros((self.layer_dims[layer], num_patterns))
                if self.inh_layer_dims[layer] > 0:
                    past_layer_inh_output_dict[layer] = np.zeros((self.inh_layer_dims[layer], num_patterns))
                    past_layer_inh_state_dict[layer] = np.zeros((self.inh_layer_dims[layer], num_patterns))
            else:
                past_layer_state_dict[layer] = np.zeros(self.layer_dims[layer])
                if self.inh_layer_dims[layer] > 0:
                    past_layer_inh_output_dict[layer] = np.zeros(self.inh_layer_dims[layer])
                    past_layer_inh_state_dict[layer] = np.zeros(self.inh_layer_dims[layer])

        for t in range(self.num_steps):
            for layer in sorted_layers:
                prev_layer_output = layer_output_dict[layer - 1]
                delta_curr_layer_input = E_E_weight_matrix_dict[layer].dot(prev_layer_output)
                if self.inh_layer_dims[layer] > 0:
                    past_layer_inh_output = past_layer_inh_output_dict[layer]
                    delta_curr_layer_input += E_I_weight_matrix_dict[layer].dot(past_layer_inh_output)
                delta_curr_layer_input -= past_layer_state_dict[layer]
                delta_curr_layer_input /= self.tau
                curr_layer_state = past_layer_state_dict[layer] + delta_curr_layer_input
                past_layer_state_dict[layer] = np.copy(curr_layer_state)
                layer_output_dict[layer] = self.activation_f(curr_layer_state)

                if self.inh_layer_dims[layer] > 0:
                    curr_layer_output = layer_output_dict[layer]
                    past_layer_inh_output = past_layer_inh_output_dict[layer]
                    delta_curr_layer_inh_input = I_E_weight_matrix_dict[layer].dot(curr_layer_output)
                    delta_curr_layer_inh_input += I_I_weight_matrix_dict[layer].dot(past_layer_inh_output)
                    delta_curr_layer_inh_input -= past_layer_inh_state_dict[layer]
                    delta_curr_layer_inh_input /= self.tau
                    curr_layer_inh_state = past_layer_inh_state_dict[layer] + delta_curr_layer_inh_input
                    past_layer_inh_state_dict[layer] = np.copy(curr_layer_inh_state)
                    layer_inh_output_dict[layer] = self.activation_f(curr_layer_inh_state)
            past_layer_inh_output_dict = deepcopy(layer_inh_output_dict)

        return layer_output_dict, layer_inh_output_dict

    def plot_network_state_summary(self, E_E_weight_matrix_dict, E_I_weight_matrix_dict, I_E_weight_matrix_dict,
                                   I_I_weight_matrix_dict, layer_output_dict, layer_inh_output_dict):
        fig, axes = plt.subplots(4, len(E_E_weight_matrix_dict),
                                 figsize=(len(E_E_weight_matrix_dict) * 2.5, 2.5 * 4))
        for col, layer in enumerate(sorted(list(E_E_weight_matrix_dict.keys()))):
            if len(E_E_weight_matrix_dict) == 1:
                this_axis = axes[0]
            else:
                this_axis = axes[0][col]
            cbar = this_axis.imshow(E_E_weight_matrix_dict[layer], aspect='auto', interpolation='none')
            fig.colorbar(cbar, ax=this_axis)
            this_axis.set_title('FF weights:\nLayer %i' % layer)
            this_axis.set_ylabel('Layer %i units' % layer)
            this_axis.set_xlabel('Layer %i units' % (layer - 1))

            if layer in E_I_weight_matrix_dict:
                if len(E_E_weight_matrix_dict) == 1:
                    this_axis = axes[1]
                else:
                    this_axis = axes[1][col]
                cbar = this_axis.imshow(E_I_weight_matrix_dict[layer], aspect='auto', interpolation='none')
                fig.colorbar(cbar, ax=this_axis)
                this_axis.set_title('E <- I weights:\nLayer %i' % layer)
                this_axis.set_ylabel('Layer %i units' % layer)
                this_axis.set_xlabel('Layer %i inh units' % (layer))

                if len(E_E_weight_matrix_dict) == 1:
                    this_axis = axes[2]
                else:
                    this_axis = axes[2][col]
                cbar = this_axis.imshow(I_E_weight_matrix_dict[layer], aspect='auto', interpolation='none')
                fig.colorbar(cbar, ax=this_axis)
                this_axis.set_title('I <- E weights:\nLayer %i' % layer)
                this_axis.set_ylabel('Layer %i inh units' % layer)
                this_axis.set_xlabel('Layer %i units' % (layer))

                if len(E_E_weight_matrix_dict) == 1:
                    this_axis = axes[3]
                else:
                    this_axis = axes[3][col]
                cbar = this_axis.imshow(I_I_weight_matrix_dict[layer], aspect='auto', interpolation='none')
                fig.colorbar(cbar, ax=this_axis)
                this_axis.set_title('I <- I weights:\nLayer %i' % layer)
                this_axis.set_ylabel('Layer %i inh units' % layer)
                this_axis.set_xlabel('Layer %i inh units' % (layer))
            else:
                for row in range(1, 4):
                    if len(E_E_weight_matrix_dict) == 1:
                        this_axis = axes[row]
                    else:
                        this_axis = axes[row][col]
                    this_axis.axis('off')
        fig.tight_layout()
        fig.show()

        fig, axes = plt.subplots(2, len(layer_output_dict),
                                 figsize=(len(layer_output_dict) * 2.5, 2.5 * 2))
        for layer in sorted(list(layer_output_dict.keys())):
            if layer == self.num_layers - 1:
                final_output = layer_output_dict[layer]
                sorted_row_indexes = get_diag_argmax_row_indexes(final_output)
                cbar = axes[0][layer].imshow(final_output[sorted_row_indexes], aspect='auto', interpolation='none')
            else:
                cbar = axes[0][layer].imshow(layer_output_dict[layer], aspect='auto', interpolation='none')
            fig.colorbar(cbar, ax=axes[0][layer])
            if layer == 0:
                axes[0][layer].set_title('Input layer\nactivities')
            elif layer == self.num_layers - 1:
                axes[0][layer].set_title('Output layer\nactivities:\nLayer %i' % layer)
            else:
                axes[0][layer].set_title('Hidden layer %i\nactivities' % layer)
            axes[0][layer].set_ylabel('Layer %i units' % layer)
            axes[0][layer].set_xlabel('Input patterns')

            if layer in layer_inh_output_dict:
                cbar = axes[1][layer].imshow(np.atleast_2d(layer_inh_output_dict[layer]), aspect='auto',
                                             interpolation='none')
                fig.colorbar(cbar, ax=axes[1][layer])
                axes[1][layer].set_title('Layer %i inh\nactivities' % layer)
                axes[1][layer].set_xlabel('Input patterns')
                axes[1][layer].set_ylabel('Layer %i inh units' % layer)

            axes[1][0].axis('off')
        fig.tight_layout()
        fig.show()

    def get_delta_weights(self, E_E_weight_matrix_dict, E_I_weight_matrix_dict, I_E_weight_matrix_dict,
                          I_I_weight_matrix_dict, layer_output_dict, layer_inh_output_dict):
        delta_E_E_weight_matrix_dict = {}
        delta_E_I_weight_matrix_dict = {}
        delta_I_E_weight_matrix_dict = {}
        delta_I_I_weight_matrix_dict = {}

        sorted_layers = sorted(list(E_E_weight_matrix_dict.keys()))

        for layer in sorted_layers:
            curr_layer_output = layer_output_dict[layer]
            prev_layer_output = layer_output_dict[layer - 1]
            delta_E_E_weight_matrix_dict[layer] = \
                self.E_E_learning_rule_dict[layer](prev_layer_output, curr_layer_output)

            if self.inh_layer_dims[layer] > 0:
                curr_layer_inh_output = layer_inh_output_dict[layer]
                delta_E_I_weight_matrix_dict[layer] = \
                    self.E_I_learning_rule_dict[layer](curr_layer_inh_output, curr_layer_output)
                delta_I_E_weight_matrix_dict[layer] = \
                    self.I_E_learning_rule_dict[layer](curr_layer_output, curr_layer_inh_output)
                delta_I_I_weight_matrix_dict[layer] = \
                    self.I_I_learning_rule_dict[layer](curr_layer_inh_output, curr_layer_inh_output)

        return delta_E_E_weight_matrix_dict, delta_E_I_weight_matrix_dict, delta_I_E_weight_matrix_dict, \
               delta_I_I_weight_matrix_dict

    def train(self, num_blocks, shuffle=False, disp=False, plot=False):
        E_E_weight_matrix_dict = deepcopy(self.initial_E_E_weight_matrix_dict)
        E_I_weight_matrix_dict = deepcopy(self.initial_E_I_weight_matrix_dict)
        I_E_weight_matrix_dict = deepcopy(self.initial_I_E_weight_matrix_dict)
        I_I_weight_matrix_dict = deepcopy(self.initial_I_I_weight_matrix_dict)

        self.E_E_weight_matrix_dict_history = []
        self.E_I_weight_matrix_dict_history = []
        self.I_E_weight_matrix_dict_history = []
        self.I_I_weight_matrix_dict_history = []

        self.input_pattern_index_history = []
        self.layer_output_dict_history = []
        self.layer_inh_output_dict_history = []

        self.block_output_activity_history = []
        self.local_accuracy_history = []
        self.global_accuracy_history = []

        num_patterns = self.input_pattern_matrix.shape[-1]
        if shuffle:
            self.random = np.random.default_rng(seed=[0, self.seed])
        for block in range(num_blocks):
            input_pattern_indexes = np.arange(num_patterns)
            if shuffle:
                self.random.shuffle(input_pattern_indexes)
            for input_pattern_index in input_pattern_indexes:
                self.input_pattern_index_history.append(input_pattern_index)
                input_pattern = self.input_pattern_matrix[:, input_pattern_index]
                target_output = self.target_output_pattern_matrix[:, input_pattern_index]

                layer_output_dict, layer_inh_output_dict = \
                    self.get_layer_activities(input_pattern, E_E_weight_matrix_dict, E_I_weight_matrix_dict,
                                                 I_E_weight_matrix_dict, I_I_weight_matrix_dict)

                delta_E_E_weight_matrix_dict, delta_E_I_weight_matrix_dict, delta_I_E_weight_matrix_dict, \
                delta_I_I_weight_matrix_dict = \
                    self.get_delta_weights(E_E_weight_matrix_dict, E_I_weight_matrix_dict, I_E_weight_matrix_dict,
                                           I_I_weight_matrix_dict, layer_output_dict, layer_inh_output_dict)

                for delta_weight_matrix_dict, weight_matrix_dict, weight_bounds_dict in \
                        zip([delta_E_E_weight_matrix_dict, delta_E_I_weight_matrix_dict, delta_I_E_weight_matrix_dict,
                             delta_I_I_weight_matrix_dict],
                            [E_E_weight_matrix_dict, E_I_weight_matrix_dict, I_E_weight_matrix_dict,
                             I_I_weight_matrix_dict],
                            [self.E_E_weight_bounds_dict, self.E_I_weight_bounds_dict, self.I_E_weight_bounds_dict,
                             self.I_I_weight_bounds_dict]):
                    for layer in delta_weight_matrix_dict:
                        this_weight_matrix = weight_matrix_dict[layer] + delta_weight_matrix_dict[layer]
                        if weight_bounds_dict[0] is not None:
                            this_weight_matrix = np.maximum(this_weight_matrix, weight_bounds_dict[0])
                        if weight_bounds_dict[1] is not None:
                            this_weight_matrix = np.minimum(this_weight_matrix, weight_bounds_dict[1])
                        weight_matrix_dict[layer] = this_weight_matrix

                for layer in I_I_weight_matrix_dict:
                    # no autapses
                    diag_indexes = np.diag_indices_from(I_I_weight_matrix_dict[layer])
                    I_I_weight_matrix_dict[layer][diag_indexes] = 0.

                E_E_weight_matrix_dict, E_I_weight_matrix_dict, I_E_weight_matrix_dict, I_I_weight_matrix_dict = \
                    self.normalize_weights(E_E_weight_matrix_dict, E_I_weight_matrix_dict, I_E_weight_matrix_dict,
                                           I_I_weight_matrix_dict)

                self.layer_output_dict_history.append(deepcopy(layer_output_dict))
                self.layer_inh_output_dict_history.append(deepcopy(layer_inh_output_dict))

                self.E_E_weight_matrix_dict_history.append(deepcopy(E_E_weight_matrix_dict))
                self.E_I_weight_matrix_dict_history.append(deepcopy(E_I_weight_matrix_dict))
                self.I_E_weight_matrix_dict_history.append(deepcopy(I_E_weight_matrix_dict))
                self.I_I_weight_matrix_dict_history.append(deepcopy(I_I_weight_matrix_dict))

            layer_output_dict, layer_inh_output_dict = \
                self.get_layer_activities(self.input_pattern_matrix, E_E_weight_matrix_dict, E_I_weight_matrix_dict,
                                          I_E_weight_matrix_dict, I_I_weight_matrix_dict)
            self.block_output_activity_history.append(np.copy(layer_output_dict[self.num_layers - 1]))

        final_output_activity = self.block_output_activity_history[-1]
        final_sorted_row_indexes = get_diag_argmax_row_indexes(final_output_activity)
        target_argmax = np.arange(num_patterns)
        for output_activity in self.block_output_activity_history:
            global_accuracy = np.count_nonzero(
                np.argmax(output_activity[final_sorted_row_indexes, :], axis=1) == target_argmax) / \
                              num_patterns * 100.
            self.global_accuracy_history.append(global_accuracy)
            local_sorted_row_indexes = get_diag_argmax_row_indexes(output_activity)
            local_accuracy = np.count_nonzero(
                np.argmax(output_activity[local_sorted_row_indexes, :], axis=1) == target_argmax) / \
                              num_patterns * 100.
            self.local_accuracy_history.append(local_accuracy)

        self.local_accuracy_history = np.array(self.local_accuracy_history)
        self.global_accuracy_history = np.array(self.global_accuracy_history)

        if plot:
            sorted_indexes = []
            start_index = 0
            for i in range(num_blocks):
                sorted_indexes.extend(
                    np.add(start_index, np.argsort(self.input_pattern_index_history[
                                                   start_index:start_index + num_patterns])))
                start_index += num_patterns
            sorted_indexes = np.array(sorted_indexes)

            fig, axes = plt.subplots(self.num_layers, figsize=(8., self.num_layers * 2.))
            cbar = axes[self.num_layers - 1].imshow(
                np.column_stack([self.input_pattern_matrix[:, self.input_pattern_index_history[i]]
                                 for i in sorted_indexes]), aspect='auto', interpolation='none')
            fig.colorbar(cbar, ax=axes[self.num_layers - 1], pad=0.02)
            axes[self.num_layers - 1].set_ylabel('Input units')
            axes[self.num_layers - 1].set_xlabel('Sorted input patterns')
            axes[self.num_layers - 1].set_title('Input layer activities')
            for row, layer in enumerate(range(self.num_layers - 1, 0, -1)):
                cbar = axes[row].imshow(
                    np.column_stack([self.layer_output_dict_history[i][layer] for i in sorted_indexes]),
                    aspect='auto', interpolation='none')
                fig.colorbar(cbar, ax=axes[row], pad=0.02)
                axes[row].set_ylabel('Layer %i units' % layer)
                # axes[row].set_xlabel('Sorted input patterns')
                axes[row].set_title('Layer %i activities' % layer)
            fig.tight_layout()
            fig.show()

            fig, axis = plt.subplots()
            axis.plot(range(num_blocks), self.global_accuracy_history, label='Global')
            axis.plot(range(num_blocks), self.local_accuracy_history, label='Local')
            axis.set_xlabel('Training blocks')
            axis.set_ylabel('Argmax accuracy')
            axis.legend(loc='best', frameon=False)
            fig.tight_layout()
            fig.show()

        return self.get_MSE_loss(self.E_E_weight_matrix_dict_history[-1], self.E_I_weight_matrix_dict_history[-1],
                                 self.I_E_weight_matrix_dict_history[-1], self.I_I_weight_matrix_dict_history[-1],
                                 disp)

    def get_MSE_loss(self, E_E_weight_matrix_dict, E_I_weight_matrix_dict, I_E_weight_matrix_dict,
                     I_I_weight_matrix_dict, disp=False):
        layer_output_dict, layer_inh_output_dict = \
            self.get_layer_activities(self.input_pattern_matrix, E_E_weight_matrix_dict, E_I_weight_matrix_dict,
                                      I_E_weight_matrix_dict, I_I_weight_matrix_dict)

        final_output = layer_output_dict[self.num_layers - 1]
        sorted_row_indexes = get_diag_argmax_row_indexes(final_output)
        loss = np.mean((self.target_output_pattern_matrix - final_output[sorted_row_indexes, :])**2.)
        if disp:
            print('Loss: %.4E, Argmax: %s' % (loss, np.argmax(final_output[sorted_row_indexes], axis=1)))
            sys.stdout.flush()
        return loss


def main():

    ReLU = lambda x: np.maximum(0., x)

    input_dim = 7
    num_hidden_layers = 0
    hidden_dim = 7
    hidden_inh_dim = 7
    output_dim = 21
    output_inh_dim = 1  #7
    tau = 3
    num_steps = 12
    seed = 0
    disp = True
    shuffle = True
    plot = 2
    n_hot = 2
    I_floor_weight = 0.  #  -0.05
    anti_Hebb_I = False

    num_blocks = 200  # each block contains all input patterns

    E_E_learning_rate = 0.05
    E_I_learning_rate = 0.05
    I_E_learning_rate = 0.05
    I_I_learning_rate = 0.05

    network = Hebb_lat_inh_network(num_hidden_layers=num_hidden_layers, input_dim=input_dim, hidden_dim=hidden_dim,
                                   hidden_inh_dim=hidden_inh_dim, output_dim=output_dim, output_inh_dim=output_inh_dim,
                                   tau=tau, num_steps=num_steps, activation_f=ReLU, seed=seed)

    input_peak_rate = 1.
    input_pattern_matrix = get_n_hot_patterns(n_hot, input_dim).T * input_peak_rate
    unit_peak_col_indexes = np.argmax(input_pattern_matrix, axis=1)
    sorted_row_indexes = np.argsort(unit_peak_col_indexes)
    input_pattern_matrix = input_pattern_matrix[sorted_row_indexes,:]
    num_patterns = input_pattern_matrix.shape[1]

    target_output_pattern_matrix = np.zeros([output_dim,num_patterns])
    output_indexes = np.arange(0, num_patterns)
    for i in output_indexes:
        target_output_pattern_matrix[i,i] = input_peak_rate

    network.load_input_patterns(input_pattern_matrix)
    network.load_target_output_patterns(target_output_pattern_matrix)

    if plot > 1:
        fig, axes = plt.subplots(1, 2, figsize=(7., 3.))

        cbar = axes[0].imshow(input_pattern_matrix, aspect='auto', interpolation='none', cmap='binary')
        fig.colorbar(cbar, ax=axes[0])
        axes[0].set_xlabel('Input pattern')
        axes[0].set_ylabel('Input unit')
        axes[0].set_title('Input activities')

        cbar = axes[1].imshow(target_output_pattern_matrix, aspect='auto', interpolation='none', cmap='binary')
        fig.colorbar(cbar, ax=axes[1])
        axes[1].set_xlabel('Input pattern')
        axes[1].set_ylabel('Output unit')
        axes[1].set_title('Target output activities')

        fig.tight_layout()
        fig.show()

    hidden_E_E_weight_scale = 2.
    hidden_E_I_weight_scale = 2.
    hidden_I_E_weight_scale = 2.
    hidden_I_I_weight_scale = 2.

    output_E_E_weight_scale = 5.
    output_E_I_weight_scale = 2.
    output_I_E_weight_scale = 5.
    output_I_I_weight_scale = 5.

    E_E_learning_rate_dict = {}
    E_I_learning_rate_dict = {}
    I_E_learning_rate_dict = {}
    I_I_learning_rate_dict = {}

    for layer in range(1, network.num_layers):
        E_E_learning_rate_dict[layer] = E_E_learning_rate
        if network.inh_layer_dims[layer] > 0:
            E_I_learning_rate_dict[layer] = E_I_learning_rate
            I_E_learning_rate_dict[layer] = I_E_learning_rate
            I_I_learning_rate_dict[layer] = I_I_learning_rate

    E_E_learning_rule_dict = {}
    E_I_learning_rule_dict = {}
    I_E_learning_rule_dict = {}
    I_I_learning_rule_dict = {}

    for layer in range(1, network.num_layers):
        E_E_learning_rule_dict[layer] = 'Hebb + weight norm'
        if network.inh_layer_dims[layer] > 0:
            if network.inh_layer_dims[layer] > 1:
                I_E_learning_rule_dict[layer] = 'Hebb + weight norm'
                if anti_Hebb_I:
                    E_I_learning_rule_dict[layer] = 'Anti-Hebb + weight norm'
                    I_I_learning_rule_dict[layer] = 'Anti-Hebb + weight norm'
                else:
                    E_I_learning_rule_dict[layer] = 'Hebb + weight norm'
                    I_I_learning_rule_dict[layer] = 'Hebb + weight norm'
            else:
                I_E_learning_rule_dict[layer] = None
                E_I_learning_rule_dict[layer] = None
                I_I_learning_rule_dict[layer] = None

    network.config_learning_rules(E_E_learning_rule_dict, E_I_learning_rule_dict, I_E_learning_rule_dict,
                                  I_I_learning_rule_dict, E_E_learning_rate_dict, E_I_learning_rate_dict,
                                  I_E_learning_rate_dict, I_I_learning_rate_dict)

    E_E_weight_scale_dict = {}
    E_I_weight_scale_dict = {}
    I_E_weight_scale_dict = {}
    I_I_weight_scale_dict = {}

    for layer in range(1, network.num_layers):
        if layer < network.num_layers - 1:
            E_E_weight_scale_dict[layer] = hidden_E_E_weight_scale
            if network.inh_layer_dims[layer] > 0:
                E_I_weight_scale_dict[layer] = hidden_E_I_weight_scale
                I_E_weight_scale_dict[layer] = hidden_I_E_weight_scale
                I_I_weight_scale_dict[layer] = hidden_I_I_weight_scale
        else:
            E_E_weight_scale_dict[layer] = output_E_E_weight_scale
            if network.inh_layer_dims[layer] > 0:
                E_I_weight_scale_dict[layer] = output_E_I_weight_scale
                I_E_weight_scale_dict[layer] = output_I_E_weight_scale
                I_I_weight_scale_dict[layer] = output_I_I_weight_scale

    network.init_weights(E_E_weight_scale_dict, E_I_weight_scale_dict, I_E_weight_scale_dict, I_I_weight_scale_dict,
                         E_I_weight_bounds_dict=(None, I_floor_weight), I_I_weight_bounds_dict=(None, I_floor_weight))

    layer_output_dict, layer_inh_output_dict = \
        network.get_layer_activities(network.input_pattern_matrix, network.initial_E_E_weight_matrix_dict,
                                     network.initial_E_I_weight_matrix_dict, network.initial_I_E_weight_matrix_dict,
                                     network.initial_I_I_weight_matrix_dict)
    if plot > 0:
        network.plot_network_state_summary(network.initial_E_E_weight_matrix_dict,
                                           network.initial_E_I_weight_matrix_dict,
                                           network.initial_I_E_weight_matrix_dict,
                                           network.initial_I_I_weight_matrix_dict,
                                           layer_output_dict, layer_inh_output_dict)

    loss = network.train(num_blocks, shuffle=shuffle, disp=disp, plot=plot > 0)

    layer_output_dict, layer_inh_output_dict = \
        network.get_layer_activities(network.input_pattern_matrix, network.E_E_weight_matrix_dict_history[-1],
                                     network.E_I_weight_matrix_dict_history[-1],
                                     network.I_E_weight_matrix_dict_history[-1],
                                     network.I_I_weight_matrix_dict_history[-1])

    if plot > 0:
        network.plot_network_state_summary(network.E_E_weight_matrix_dict_history[-1],
                                           network.E_I_weight_matrix_dict_history[-1],
                                           network.I_E_weight_matrix_dict_history[-1],
                                           network.I_I_weight_matrix_dict_history[-1], layer_output_dict,
                                           layer_inh_output_dict)

    plt.show()

    context.update(locals())


if __name__ == '__main__':
    main()