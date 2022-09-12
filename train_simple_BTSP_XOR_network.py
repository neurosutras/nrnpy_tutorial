import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from copy import deepcopy
from nested.optimize_utils import Context
from train_network_utils import *


context = Context()


def scaled_single_sigmoid(th, peak, x=None, ylim=None):
    """
    Transform a sigmoid to intersect x and y range limits.
    :param th: float
    :param peak: float
    :param x: array
    :param ylim: pair of float
    :return: callable
    """
    if x is None:
        x = (0., 1.)
    if ylim is None:
        ylim = (0., 1.)
    if th < x[0] or th > x[-1]:
        raise ValueError('scaled_single_sigmoid: th: %.2E is out of range for xlim: [%.2E, %.2E]' % (th, x[0], x[-1]))
    if peak == th:
        raise ValueError('scaled_single_sigmoid: peak and th: %.2E cannot be equal' % th)
    slope = 2. / (peak - th)
    y = lambda x: 1. / (1. + np.exp(-slope * (x - th)))
    start_val = y(x[0])
    end_val = y(x[-1])
    amp = end_val - start_val
    target_amp = ylim[1] - ylim[0]
    return lambda xi: (target_amp / amp) * (1. / (1. + np.exp(-slope * (xi - th))) - start_val) + ylim[0]


def get_BTSP_delta_w(learning_rate, dep_ratio, dep_th, dep_width):
    """
    TODO: vectorize
    :param learning_rate:
    :param dep_ratio:
    :param dep_th:
    :param dep_width:
    :return:
    """
    f_dep = scaled_single_sigmoid(dep_th, dep_th + dep_width)
    return np.vectorize(lambda pre, w, mod, w_max: learning_rate * np.abs(mod) * (
            (w_max - w) * np.maximum(0., np.minimum(pre, 1.)) - w * dep_ratio *
            f_dep(np.maximum(0., np.minimum(pre, 1.)))), excluded=['w_max'])


def get_neg_mod_delta_w(learning_rate):
    return np.vectorize(lambda pre: -learning_rate * np.maximum(0., np.minimum(pre, 1.)))


def get_Hebb_rule(learning_rate, direction=1):
    return lambda pre, post: direction * learning_rate * np.outer(post, pre)


def get_anti_Hebb_rule(learning_rate, direction=1):
    return lambda pre, post: -direction * learning_rate * np.outer(post, pre)


class BTSP_network(object):

    def __init__(self, num_hidden_layers, input_dim, hidden_dim, hidden_inh_soma_dim, hidden_inh_dend_dim, output_dim,
                 output_inh_soma_dim, tau, num_steps, activation_f=None, seed=None):
        self.num_hidden_layers = num_hidden_layers
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.hidden_inh_soma_dim = hidden_inh_soma_dim
        self.hidden_inh_dend_dim = hidden_inh_dend_dim
        self.output_dim = output_dim
        self.output_inh_soma_dim = output_inh_soma_dim
        self.tau = tau
        self.num_steps = num_steps
        self.activation_f = activation_f
        self.seed = seed
        self.random = np.random.default_rng(seed=self.seed)
        self.num_layers = num_hidden_layers + 2
        if num_hidden_layers > 0:
            if isinstance(self.hidden_dim, list):
                if len(self.hidden_dim) != self.num_hidden_layers:
                    raise Exception('BTSP_network: hidden_dim must be int or list of len num_hidden_layers')
                self.layer_dims = [self.input_dim] + self.hidden_dim + [self.output_dim]

                if not isinstance(self.hidden_inh_soma_dim, list) or \
                        len(self.hidden_inh_soma_dim) != self.num_hidden_layers:
                    raise Exception('BTSP_network: hidden_inh_soma_dim must be int or list of len num_hidden_layers')
                self.inh_soma_layer_dims = [0] + self.hidden_inh_soma_dim + [self.output_inh_soma_dim]

                if not isinstance(self.hidden_inh_dend_dim, list) or \
                        len(self.hidden_inh_dend_dim) != self.num_hidden_layers:
                    raise Exception('BTSP_network: hidden_inh_dend_dim must be int or list of len num_hidden_layers')
                self.inh_dend_layer_dims = [0] + self.hidden_inh_dend_dim + [0]

            elif isinstance(self.hidden_dim, int):
                self.layer_dims = [self.input_dim] + self.num_hidden_layers * [self.hidden_dim] + [self.output_dim]

                if not isinstance(self.hidden_inh_soma_dim, int):
                    raise Exception('BTSP_network: hidden_inh_soma_dim must be int or list of len num_hidden_layers')
                self.inh_soma_layer_dims = [0] + self.num_hidden_layers * [self.hidden_inh_soma_dim] + \
                                           [self.output_inh_soma_dim]

                if not isinstance(self.hidden_inh_dend_dim, int):
                    raise Exception('BTSP_network: hidden_inh_dend_dim must be int or list of len num_hidden_layers')
                self.inh_dend_layer_dims = [0] + self.num_hidden_layers * [self.hidden_inh_dend_dim] + \
                                           [0]
        else:
            self.layer_dims = [self.input_dim, self.output_dim]
            self.inh_soma_layer_dims = [0, self.output_inh_soma_dim]
            self.inh_dend_layer_dims = [0, 0]

    def init_weights(self, FF_weight_bounds_dict, FB_weight_bounds_dict, E_I_soma_weight_bounds, I_soma_E_weight_bounds,
                     initial_FF_weight_bounds_dict, initial_FB_weight_bounds_dict, initial_I_soma_E_weight_bounds,
                     initial_I_dend_E_weight_bounds, initial_E_I_dend_weight_bounds, initial_E_I_soma_weight_bounds,
                     initial_E_bias_bounds):
        self.FF_weight_bounds_dict = FF_weight_bounds_dict
        self.FB_weight_bounds_dict = FB_weight_bounds_dict
        self.E_I_soma_weight_bounds = E_I_soma_weight_bounds
        self.I_soma_E_weight_bounds = I_soma_E_weight_bounds
        self.initial_FF_weight_bounds_dict = initial_FF_weight_bounds_dict
        self.initial_FB_weight_bounds_dict = initial_FB_weight_bounds_dict
        self.initial_I_soma_E_weight_bounds = initial_I_soma_E_weight_bounds
        self.initial_I_dend_E_weight_bounds = initial_I_dend_E_weight_bounds
        self.initial_E_I_dend_weight_bounds = initial_E_I_dend_weight_bounds
        self.initial_E_I_soma_weight_bounds = initial_E_I_soma_weight_bounds
        self.initial_E_bias_bounds = initial_E_bias_bounds
        self.re_init_weights()

    def re_init_weights(self, seed=None):
        if seed is not None:
            self.seed = seed
        self.random = np.random.default_rng(seed=self.seed)
        self.initial_FF_weight_matrix_dict = {}
        self.initial_FB_weight_matrix_dict = {}
        self.initial_I_soma_E_weight_matrix_dict = {}
        self.initial_I_dend_E_weight_matrix_dict = {}
        self.initial_E_I_soma_weight_matrix_dict = {}
        self.initial_E_I_dend_weight_matrix_dict = {}
        self.initial_E_bias_array_dict = {}

        for layer in range(1, self.num_layers):
            curr_layer_dim = self.layer_dims[layer]
            prev_layer_dim = self.layer_dims[layer - 1]
            self.initial_FF_weight_matrix_dict[layer] = \
                self.random.uniform(self.initial_FF_weight_bounds_dict[layer][0],
                                    self.initial_FF_weight_bounds_dict[layer][1],
                                    [curr_layer_dim, prev_layer_dim])
            self.initial_E_bias_array_dict[layer] = np.ones([curr_layer_dim]) * self.initial_E_bias_bounds[layer][0]
            curr_layer_inh_soma_dim = self.inh_soma_layer_dims[layer]
            if curr_layer_inh_soma_dim > 0:
                if curr_layer_inh_soma_dim > 1:
                    raise Exception('curr_layer_inh_soma_dim > 1 not yet implemented')
                    self.initial_I_soma_E_weight_matrix_dict[layer] = \
                        self.random.uniform(self.initial_I_soma_E_weight_bounds[layer][0],
                                            self.initial_I_soma_E_weight_bounds[layer][1],
                                            [curr_layer_inh_soma_dim, curr_layer_dim])
                    self.initial_E_I_soma_weight_matrix_dict[layer] = \
                        self.random.uniform(self.initial_E_I_soma_weight_bounds[layer][0],
                                            self.initial_E_I_soma_weight_bounds[layer][1],
                                            [curr_layer_dim, curr_layer_inh_soma_dim])
                else:
                    self.initial_I_soma_E_weight_matrix_dict[layer] = \
                        np.ones([curr_layer_inh_soma_dim, curr_layer_dim]) * \
                        self.initial_I_soma_E_weight_bounds[layer][0]
                    self.initial_E_I_soma_weight_matrix_dict[layer] = \
                        np.ones([curr_layer_dim, curr_layer_inh_soma_dim]) * \
                        self.initial_E_I_soma_weight_bounds[layer][0]

        for layer in range(1, self.num_layers - 1):
            curr_layer_dim = self.layer_dims[layer]
            next_layer_dim = self.layer_dims[layer + 1]
            self.initial_FB_weight_matrix_dict[layer] = \
                self.random.uniform(self.initial_FB_weight_bounds_dict[layer][0],
                                    self.initial_FB_weight_bounds_dict[layer][1],
                                    [curr_layer_dim, next_layer_dim])
            curr_layer_inh_dend_dim = self.inh_dend_layer_dims[layer]
            if curr_layer_inh_dend_dim > 0:
                if curr_layer_inh_dend_dim > 1:
                    raise Exception('curr_layer_inh_dend_dim > 1 not yet implemented')
                    self.initial_I_dend_E_weight_matrix_dict[layer] = \
                        self.random.uniform(self.initial_I_dend_E_weight_bounds[layer][0],
                                            self.initial_I_dend_E_weight_bounds[layer][1],
                                            [curr_layer_inh_dend_dim, curr_layer_dim])
                    self.initial_E_I_dend_weight_matrix_dict[layer] = \
                        self.random.uniform(self.initial_E_I_dend_weight_bounds[layer][0],
                                            self.initial_E_I_dend_weight_bounds[layer][1],
                                            [curr_layer_dim, curr_layer_inh_dend_dim])
                else:
                    """
                    TODO: Implement optional learning on E_I_dend weights 
                    """
                    self.initial_I_dend_E_weight_matrix_dict[layer] = \
                        np.ones([curr_layer_inh_dend_dim, curr_layer_dim]) * \
                        self.initial_I_dend_E_weight_bounds[layer][0]
                    self.initial_E_I_dend_weight_matrix_dict[layer] = \
                        np.ones([curr_layer_dim, curr_layer_inh_dend_dim]) * \
                        self.initial_E_I_dend_weight_bounds[layer][0]

    def load_input_patterns(self, input_pattern_matrix):
        if input_pattern_matrix.shape[0] != self.input_dim:
            raise Exception('BTSP_network.load_input_patterns: input_pattern_matrix shape does not match input_dim')
        if hasattr(self, 'target_output_pattern_matrix') and \
                self.target_output_pattern_matrix.shape[-1] != input_pattern_matrix.shape[-1]:
            raise Exception('BTSP_network.load_input_patterns: input_pattern_matrix shape does not match '
                            'target_output_pattern_matrix')
        self.input_pattern_matrix = input_pattern_matrix
        
    def load_target_output_patterns(self, target_output_pattern_matrix):
        if target_output_pattern_matrix.shape[0] != self.output_dim:
            raise Exception('BTSP_network.load_target_output_patterns: target_output_pattern_matrix shape does not '
                            'match output_dim')
        if hasattr(self, 'input_pattern_matrix') and \
                self.input_pattern_matrix.shape[-1] != target_output_pattern_matrix.shape[-1]:
            raise Exception('BTSP_network.load_target_output_patterns: input_pattern_matrix shape does not match '
                            'target_output_pattern_matrix')
        self.target_output_pattern_matrix = target_output_pattern_matrix

    def config_BTSP_rule(self, pos_mod_learning_rate_dict, pos_mod_th_dict,
                         neg_mod_th_dict, dep_ratio, dep_th, dep_width, neg_mod_pre_discount):
        self.pos_mod_learning_rate_dict = pos_mod_learning_rate_dict
        self.pos_mod_th_dict = pos_mod_th_dict
        self.neg_mod_th_dict = neg_mod_th_dict
        self.pos_mod_rule_dict = {}
        self.neg_mod_pre_discount = neg_mod_pre_discount
        for layer in pos_mod_learning_rate_dict:
            self.pos_mod_rule_dict[layer] = \
                get_BTSP_delta_w(pos_mod_learning_rate_dict[layer], dep_ratio, dep_th, dep_width)

    def config_bias_learning_rule(self, E_bias_learning_rate_dict):
        self.E_bias_learning_rate_dict = E_bias_learning_rate_dict

    def config_E_I_soma_learning_rule(self, E_I_soma_learning_rate_dict):
        self.E_I_soma_learning_rate_dict = E_I_soma_learning_rate_dict

    def config_I_soma_E_learning_rule(self, I_soma_E_learning_rate_dict, I_soma_E_learning_rule_dict,
                                      I_soma_E_weight_scale_dict):
        self.I_soma_E_learning_rate_dict = I_soma_E_learning_rate_dict
        self.I_soma_E_weight_scale_dict = I_soma_E_weight_scale_dict
        self.I_soma_E_weight_norm_dict = {}
        self.I_soma_E_learning_rule_dict = {}
        for layer in I_soma_E_learning_rate_dict:
            if I_soma_E_learning_rule_dict[layer] == 'Hebb + weight norm':
                self.I_soma_E_learning_rule_dict[layer] = \
                    get_Hebb_rule(I_soma_E_learning_rate_dict[layer], direction=1)
                self.I_soma_E_weight_norm_dict[layer] = True
            elif I_soma_E_learning_rule_dict[layer] == 'Anti-Hebb + weight norm':
                self.I_soma_E_learning_rule_dict[layer] = \
                    get_anti_Hebb_rule(I_soma_E_learning_rate_dict[layer], direction=1)
                self.I_soma_E_weight_norm_dict[layer] = True

    def get_layer_activities(self, input_pattern_matrix, FF_weight_matrix_dict, FB_weight_matrix_dict, 
                             I_soma_E_weight_matrix_dict, I_dend_E_weight_matrix_dict, E_I_soma_weight_matrix_dict,
                             E_I_dend_weight_matrix_dict, E_bias_array_dict):
        
        summed_FF_input_dict = {}
        summed_FB_input_dict = {}

        layer_activity_dict = {}
        layer_activity_dict[0] = np.copy(input_pattern_matrix)
        prev_layer_activity = input_pattern_matrix
        sorted_layers = sorted(list(FF_weight_matrix_dict.keys()))

        layer_inh_soma_activity_dict = {}
        layer_inh_dend_activity_dict = {}
        past_layer_inh_soma_activity_dict = {}
        past_layer_state_dict = {}
        past_layer_inh_soma_state_dict = {}

        for layer in sorted_layers:
            if len(input_pattern_matrix.shape) > 1:
                num_patterns = input_pattern_matrix.shape[-1]
            else:
                num_patterns = 1
            past_layer_state_dict[layer] = np.zeros((self.layer_dims[layer], num_patterns))
            if self.inh_soma_layer_dims[layer] > 0:
                past_layer_inh_soma_activity_dict[layer] = np.zeros((self.inh_soma_layer_dims[layer], num_patterns))
                past_layer_inh_soma_state_dict[layer] = np.zeros((self.inh_soma_layer_dims[layer], num_patterns))

        for t in range(self.num_steps):
            for layer in sorted_layers:
                prev_layer_activity = layer_activity_dict[layer - 1]
                delta_curr_layer_input = FF_weight_matrix_dict[layer].dot(prev_layer_activity)
                delta_curr_layer_input += E_bias_array_dict[layer][:,np.newaxis]
                if self.inh_soma_layer_dims[layer] > 0:
                    delta_curr_layer_input += \
                        E_I_soma_weight_matrix_dict[layer].dot(past_layer_inh_soma_activity_dict[layer])
                delta_curr_layer_input -= past_layer_state_dict[layer]
                delta_curr_layer_input /= self.tau
                curr_layer_state = past_layer_state_dict[layer] + delta_curr_layer_input
                past_layer_state_dict[layer] = np.copy(curr_layer_state)
                layer_activity_dict[layer] = self.activation_f(curr_layer_state)

                if self.inh_soma_layer_dims[layer] > 0:
                    curr_layer_activity = layer_activity_dict[layer]
                    delta_curr_layer_inh_soma_input = I_soma_E_weight_matrix_dict[layer].dot(curr_layer_activity)
                    delta_curr_layer_inh_soma_input -= past_layer_inh_soma_state_dict[layer]
                    delta_curr_layer_inh_soma_input /= self.tau
                    curr_layer_inh_soma_state = past_layer_inh_soma_state_dict[layer] + delta_curr_layer_inh_soma_input
                    past_layer_inh_soma_state_dict[layer] = np.copy(curr_layer_inh_soma_state)
                    layer_inh_soma_activity_dict[layer] = self.activation_f(curr_layer_inh_soma_state)
            past_layer_inh_soma_activity_dict = deepcopy(layer_inh_soma_activity_dict)

        for layer in sorted_layers:
            prev_layer_activity = layer_activity_dict[layer - 1]
            summed_FF_input = FF_weight_matrix_dict[layer].dot(prev_layer_activity)
            summed_FF_input_dict[layer] = np.copy(summed_FF_input)

            curr_layer_activity = layer_activity_dict[layer]
            if self.inh_dend_layer_dims[layer] > 0:
                curr_layer_inh_dend_input = I_dend_E_weight_matrix_dict[layer].dot(curr_layer_activity)
                layer_inh_dend_activity_dict[layer] = self.activation_f(curr_layer_inh_dend_input)

        for layer in sorted(list(FB_weight_matrix_dict), reverse=True):
            next_layer_activity = layer_activity_dict[layer + 1]
            summed_FB_input = FB_weight_matrix_dict[layer].dot(next_layer_activity)
            summed_FB_input_dict[layer] = np.copy(summed_FB_input)

        return summed_FF_input_dict, summed_FB_input_dict, layer_activity_dict, layer_inh_soma_activity_dict, \
               layer_inh_dend_activity_dict

    def plot_network_state_summary(self, FF_weight_matrix_dict, FB_weight_matrix_dict, summed_FF_input_dict,
                                   summed_FB_input_dict, layer_activity_dict, layer_inh_soma_activity_dict):
        fig, axes = plt.subplots(2, len(FF_weight_matrix_dict),
                                 figsize=(len(FF_weight_matrix_dict) * 2.5, 2.5 * 2))
        for col, layer in enumerate(sorted(list(FF_weight_matrix_dict.keys()))):
            if len(FF_weight_matrix_dict) == 1:
                this_axis = axes[0]
            else:
                this_axis = axes[0][col]
                axes[1][-1].axis('off')
            cbar = this_axis.imshow(FF_weight_matrix_dict[layer], aspect='auto', interpolation='none')
            fig.colorbar(cbar, ax=this_axis)
            this_axis.set_title('FF weights:\nLayer %i' % layer)
            this_axis.set_ylabel('Layer %i units' % layer)
            this_axis.set_xlabel('Layer %i units' % (layer - 1))

            if layer in FB_weight_matrix_dict:
                cbar = axes[1][col].imshow(FB_weight_matrix_dict[layer], aspect='auto', interpolation='none')
                fig.colorbar(cbar, ax=axes[1][col])
                axes[1][col].set_title('FB weights:\nLayer %i' % layer)
                axes[1][col].set_ylabel('Layer %i units' % layer)
                axes[1][col].set_xlabel('Layer %i units' % (layer + 1))
        fig.tight_layout()
        fig.show()

        fig, axes = plt.subplots(3, len(layer_activity_dict),
                                 figsize=(len(layer_activity_dict) * 2.5, 2.5 * 3))
        for layer in sorted(list(layer_activity_dict.keys())):
            cbar = axes[0][layer].imshow(layer_activity_dict[layer], aspect='auto', interpolation='none')
            fig.colorbar(cbar, ax=axes[0][layer])
            if layer == 0:
                axes[0][layer].set_title('Input layer\nactivities')
            else:
                axes[0][layer].set_title('Output activities:\nLayer %i' % layer)
            axes[0][layer].set_ylabel('Layer %i units' % layer)
            axes[0][layer].set_xlabel('Input patterns')

            if layer in FB_weight_matrix_dict:
                cbar = axes[1][layer].imshow(summed_FB_input_dict[layer], aspect='auto', interpolation='none')
                fig.colorbar(cbar, ax=axes[1][layer])
                axes[1][layer].set_title('Summed FB input:\nlayer %i' % layer)
                axes[1][layer].set_ylabel('Layer %i units' % layer)
                axes[1][layer].set_xlabel('Input patterns')

            if layer in layer_inh_soma_activity_dict:
                cbar = axes[2][layer].imshow(np.atleast_2d(layer_inh_soma_activity_dict[layer]), aspect='auto',
                                             interpolation='none')
                fig.colorbar(cbar, ax=axes[2][layer])
                axes[2][layer].set_title('Inh_soma unit\nactivity: layer %i' % layer)
                axes[2][layer].set_xlabel('Input patterns')
                axes[2][layer].set_yticks([])

            axes[1][0].axis('off')
            axes[2][0].axis('off')
            axes[1][-1].axis('off')
            # axes[2][-1].axis('off')
        fig.tight_layout()
        fig.show()

    def get_layer_mod_events(self, FB_weight_matrix_dict, layer_activity_dict, I_dend_E_weight_matrix_dict,
                             E_I_dend_weight_matrix_dict, target_output):

        layer_inh_dend_activity_dict = {}
        layer_mod_events_dict = {}
        num_layers = len(layer_activity_dict)
        for layer in range(num_layers - 1, 0, -1):
            output = np.squeeze(layer_activity_dict[layer])
            mod_events = np.zeros_like(output)
            if layer == num_layers - 1:
                loss = target_output - output
                pos_mod_indexes = np.where(loss > self.pos_mod_th_dict[layer])
                mod_events[pos_mod_indexes] = np.minimum(1., loss[pos_mod_indexes])
                output[pos_mod_indexes] += mod_events[pos_mod_indexes]
                neg_mod_indexes = np.where(loss < self.neg_mod_th_dict[layer])
                mod_events[neg_mod_indexes] = np.maximum(-1., loss[neg_mod_indexes])
                # output[neg_mod_indexes] -= self.neg_mod_th_dict[layer] # = 0.
            else:
                next_layer_activity = np.squeeze(layer_activity_dict[layer + 1])
                summed_FB_input = FB_weight_matrix_dict[layer].dot(next_layer_activity)
                layer_inh_dend_activity = I_dend_E_weight_matrix_dict[layer].dot(output)
                layer_loss = summed_FB_input + E_I_dend_weight_matrix_dict[layer].dot(layer_inh_dend_activity)
                sorted_indexes = np.argsort(layer_loss)[::-1]
                for index in sorted_indexes:
                    layer_inh_dend_activity = I_dend_E_weight_matrix_dict[layer].dot(output)
                    unit_loss = summed_FB_input[index] + \
                                E_I_dend_weight_matrix_dict[layer][index].dot(layer_inh_dend_activity)
                    if unit_loss > self.pos_mod_th_dict[layer]:
                        mod_events[index] = min(1., unit_loss)
                        output[index] += mod_events[index]
                    elif unit_loss < self.neg_mod_th_dict[layer]:
                        mod_events[index] = max(-1., unit_loss)
                        # output[index] -= self.neg_mod_th_dict[layer] # = 0.
                layer_inh_dend_activity_dict[layer] = layer_inh_dend_activity
            layer_mod_events_dict[layer] = np.copy(mod_events)
            layer_activity_dict[layer] = output
        return layer_mod_events_dict, layer_activity_dict, layer_inh_dend_activity_dict

    def get_BTSP_delta_weights(self, layer_activity_dict, layer_mod_events_dict, FF_weight_matrix_dict,
                               FB_weight_matrix_dict):
        delta_FF_weight_matrix_dict = {}
        delta_FB_weight_matrix_dict = {}
        for layer in layer_mod_events_dict:
            mod = layer_mod_events_dict[layer]
            FF_pre = np.squeeze(layer_activity_dict[layer - 1])
            max_weight = self.FF_weight_bounds_dict[layer][1]
            delta_FF_weight_matrix = np.zeros_like(FF_weight_matrix_dict[layer])
            pos_mod_indexes = np.where(mod > 0.)
            if len(pos_mod_indexes[0]) > 0:
                for index in pos_mod_indexes[0]:
                    delta_FF_weight_matrix[index, :] = \
                        self.pos_mod_rule_dict[layer](FF_pre, FF_weight_matrix_dict[layer][index, :], mod[index],
                                                      max_weight)
            neg_mod_indexes = np.where(mod < 0.)
            if len(neg_mod_indexes[0]) > 0:
                for index in neg_mod_indexes[0]:
                    delta_FF_weight_matrix[index, :] = \
                        self.pos_mod_rule_dict[layer](FF_pre * self.neg_mod_pre_discount,
                                                      FF_weight_matrix_dict[layer][index, :], mod[index], max_weight)
            delta_FF_weight_matrix_dict[layer] = np.copy(delta_FF_weight_matrix)

            if layer in FB_weight_matrix_dict:
                FB_pre = np.squeeze(layer_activity_dict[layer + 1])
                max_weight = self.FB_weight_bounds_dict[layer][1]
                delta_FB_weight_matrix = np.zeros_like(FB_weight_matrix_dict[layer])
                if len(pos_mod_indexes[0]) > 0:
                    for index in pos_mod_indexes[0]:
                        delta_FB_weight_matrix[index, :] = \
                            self.pos_mod_rule_dict[layer](FB_pre, FB_weight_matrix_dict[layer][index, :], mod[index],
                                                          max_weight)
                if len(neg_mod_indexes[0]) > 0:
                    for index in neg_mod_indexes[0]:
                        delta_FB_weight_matrix[index, :] = \
                            self.pos_mod_rule_dict[layer](FB_pre * self.neg_mod_pre_discount,
                                                          FB_weight_matrix_dict[layer][index, :], mod[index],
                                                          max_weight)
                delta_FB_weight_matrix_dict[layer] = np.copy(delta_FB_weight_matrix)

        return delta_FF_weight_matrix_dict, delta_FB_weight_matrix_dict

    def train(self, num_blocks, shuffle=False, disp=False, plot=False):
        FF_weight_matrix_dict = deepcopy(self.initial_FF_weight_matrix_dict)
        FB_weight_matrix_dict = deepcopy(self.initial_FB_weight_matrix_dict)
        I_soma_E_weight_matrix_dict = deepcopy(self.initial_I_soma_E_weight_matrix_dict)
        I_dend_E_weight_matrix_dict = deepcopy(self.initial_I_dend_E_weight_matrix_dict)
        E_I_soma_weight_matrix_dict = deepcopy(self.initial_E_I_soma_weight_matrix_dict)
        E_I_dend_weight_matrix_dict = deepcopy(self.initial_E_I_dend_weight_matrix_dict)
        E_bias_array_dict = deepcopy(self.initial_E_bias_array_dict)

        self.FF_weight_matrix_dict_history = []
        self.FB_weight_matrix_dict_history = []
        self.E_I_soma_weight_matrix_dict_history = []
        self.I_soma_E_weight_matrix_dict_history = []
        self.E_I_dend_weight_matrix_dict_history = []
        self.E_bias_array_dict_history = []
        self.input_pattern_index_history = []
        self.layer_activity_dict_history = []
        self.layer_mod_events_dict_history = []
        self.block_output_activity_history = []
        self.accuracy_history = []

        num_patterns = self.input_pattern_matrix.shape[-1]
        if shuffle:
            self.random = np.random.default_rng(seed=[0, self.seed])
        for block in range(num_blocks):
            input_pattern_indexes = np.arange(num_patterns)
            if shuffle:
                self.random.shuffle(input_pattern_indexes)
            for input_pattern_index in input_pattern_indexes:
                self.input_pattern_index_history.append(input_pattern_index)
                input_pattern = np.expand_dims(self.input_pattern_matrix[:, input_pattern_index], axis=-1)
                target_output = self.target_output_pattern_matrix[:, input_pattern_index]

                summed_FF_input_dict, summed_FB_input_dict, layer_activity_dict, layer_inh_soma_activity_dict, \
                layer_inh_dend_activity_dict = \
                    self.get_layer_activities(input_pattern, FF_weight_matrix_dict, FB_weight_matrix_dict,
                                              I_soma_E_weight_matrix_dict, I_dend_E_weight_matrix_dict,
                                              E_I_soma_weight_matrix_dict, E_I_dend_weight_matrix_dict,
                                              E_bias_array_dict)
                layer_mod_events_dict, layer_activity_dict, layer_inh_dend_activity_dict = \
                    self.get_layer_mod_events(FB_weight_matrix_dict, layer_activity_dict, I_dend_E_weight_matrix_dict,
                                              E_I_dend_weight_matrix_dict, target_output)

                delta_FF_weight_matrix_dict, delta_FB_weight_matrix_dict = \
                    self.get_BTSP_delta_weights(layer_activity_dict, layer_mod_events_dict, FF_weight_matrix_dict,
                                                FB_weight_matrix_dict)

                for layer in delta_FF_weight_matrix_dict:
                    prev_FF_weight_matrix = FF_weight_matrix_dict[layer]
                    FF_weight_matrix = np.minimum(self.FF_weight_bounds_dict[layer][1],
                                                  np.maximum(self.FF_weight_bounds_dict[layer][0],
                                                             prev_FF_weight_matrix +
                                                             delta_FF_weight_matrix_dict[layer]))
                    FF_weight_matrix_dict[layer] = FF_weight_matrix
                    delta_FF_weight_matrix = FF_weight_matrix - prev_FF_weight_matrix
                    delta_FF_weight_matrix_dict[layer] = delta_FF_weight_matrix

                    delta_E_bias_array = self.E_bias_learning_rate_dict[layer] * layer_mod_events_dict[layer]
                    E_bias_array_dict[layer] += delta_E_bias_array

                    delta_E_I_soma_weight_matrix = self.E_I_soma_learning_rate_dict[layer] * \
                                                   np.outer(layer_mod_events_dict[layer],
                                                            layer_inh_soma_activity_dict[layer])
                    prev_E_I_soma_weight_matrix = E_I_soma_weight_matrix_dict[layer]
                    E_I_soma_weight_matrix_dict[layer] = np.minimum(self.E_I_soma_weight_bounds[layer][1],
                                                                    np.maximum(
                                                                        self.E_I_soma_weight_bounds[layer][0],
                                                                        prev_E_I_soma_weight_matrix +
                                                                        delta_E_I_soma_weight_matrix))

                    delta_I_soma_E_weight_matrix = \
                        self.I_soma_E_learning_rule_dict[layer](layer_activity_dict[layer],
                                                                layer_inh_soma_activity_dict[layer])
                    prev_I_soma_E_weight_matrix = I_soma_E_weight_matrix_dict[layer]
                    I_soma_E_weight_matrix_dict[layer] = np.minimum(self.I_soma_E_weight_bounds[layer][1],
                                                                    np.maximum(
                                                                        self.I_soma_E_weight_bounds[layer][0],
                                                                        prev_I_soma_E_weight_matrix +
                                                                        delta_I_soma_E_weight_matrix))
                    if self.I_soma_E_weight_norm_dict[layer]:
                        I_soma_E_weight_matrix_dict[layer] = \
                            self.I_soma_E_weight_scale_dict[layer] * I_soma_E_weight_matrix_dict[layer] / \
                                         np.sum(np.abs(I_soma_E_weight_matrix_dict[layer]), axis=1)[:, np.newaxis]

                for layer in delta_FB_weight_matrix_dict:
                    prev_FB_weight_matrix = FB_weight_matrix_dict[layer]
                    FB_weight_matrix = np.minimum(self.FB_weight_bounds_dict[layer][1],
                                                  np.maximum(self.FB_weight_bounds_dict[layer][0],
                                                             prev_FB_weight_matrix +
                                                             delta_FB_weight_matrix_dict[layer]))
                    FB_weight_matrix_dict[layer] = FB_weight_matrix
                    delta_FB_weight_matrix = FB_weight_matrix - prev_FB_weight_matrix
                    delta_FB_weight_matrix_dict[layer] = delta_FB_weight_matrix

                """
                TODO: Learn E_I_dend weights.
                """

                self.layer_activity_dict_history.append(deepcopy(layer_activity_dict))
                self.layer_mod_events_dict_history.append(deepcopy(layer_mod_events_dict))
                
                self.FF_weight_matrix_dict_history.append(deepcopy(FF_weight_matrix_dict))
                self.FB_weight_matrix_dict_history.append(deepcopy(FB_weight_matrix_dict))
                self.E_I_soma_weight_matrix_dict_history.append(deepcopy(E_I_soma_weight_matrix_dict))
                self.I_soma_E_weight_matrix_dict_history.append(deepcopy(I_soma_E_weight_matrix_dict))
                self.E_I_dend_weight_matrix_dict_history.append(deepcopy(E_I_dend_weight_matrix_dict))
                self.E_bias_array_dict_history.append(deepcopy(E_bias_array_dict))

            summed_FF_input_dict, summed_FB_input_dict, layer_activity_dict, layer_inh_soma_activity_dict, \
            layer_inh_dend_activity_dict = \
                self.get_layer_activities(self.input_pattern_matrix, FF_weight_matrix_dict, FB_weight_matrix_dict,
                                          I_soma_E_weight_matrix_dict, I_dend_E_weight_matrix_dict,
                                          E_I_soma_weight_matrix_dict, E_I_dend_weight_matrix_dict, E_bias_array_dict)
            self.block_output_activity_history.append(np.copy(layer_activity_dict[self.num_layers - 1]))
            target_argmax = np.argmax(self.target_output_pattern_matrix, axis=0)
            accuracy = np.count_nonzero(
                np.argmax(layer_activity_dict[self.num_layers - 1], axis=0) == target_argmax) / num_patterns * 100.
            self.accuracy_history.append(accuracy)

        self.accuracy_history = np.array(self.accuracy_history)

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
                    np.column_stack([self.layer_mod_events_dict_history[i][layer] for i in sorted_indexes]),
                    aspect='auto', interpolation='none')
                fig.colorbar(cbar, ax=axes[row], pad=0.02)
                axes[row].set_ylabel('Layer %i units' % layer)
                # axes[row].set_xlabel('Sorted input patterns')
                axes[row].set_title('Layer %i modulatory events' % layer)
            fig.tight_layout()
            fig.show()

            fig, axis = plt.subplots()
            axis.plot(range(num_blocks), self.accuracy_history)
            axis.set_xlabel('Training blocks')
            axis.set_ylabel('Argmax accuracy')
            fig.tight_layout()
            fig.show()

        return self.get_MSE_loss(self.FF_weight_matrix_dict_history[-1], self.FB_weight_matrix_dict_history[-1],
                                 I_soma_E_weight_matrix_dict, I_dend_E_weight_matrix_dict, E_I_soma_weight_matrix_dict,
                                 self.E_I_dend_weight_matrix_dict_history[-1], self.E_bias_array_dict_history[-1], disp)

    def get_MSE_loss(self, FF_weight_matrix_dict, FB_weight_matrix_dict, I_soma_E_weight_matrix_dict,
                     I_dend_E_weight_matrix_dict, E_I_soma_weight_matrix_dict, E_I_dend_weight_matrix_dict,
                     E_bias_array_dict, disp=False):
        summed_FF_input_dict, summed_FB_input_dict, layer_activity_dict, layer_inh_soma_activity_dict, \
        layer_inh_dend_activity_dict = \
            self.get_layer_activities(self.input_pattern_matrix, FF_weight_matrix_dict, FB_weight_matrix_dict,
                                      I_soma_E_weight_matrix_dict, I_dend_E_weight_matrix_dict,
                                      E_I_soma_weight_matrix_dict, E_I_dend_weight_matrix_dict, E_bias_array_dict)
        loss = np.mean((self.target_output_pattern_matrix - layer_activity_dict[self.num_layers - 1])**2.)
        if disp:
            print('Loss: %.4E, Argmax: %s' % (loss, np.argmax(layer_activity_dict[self.num_layers - 1], axis=0)))
            sys.stdout.flush()
        return loss


def main():

    ReLU = lambda x: np.maximum(0., x)

    input_dim = 2
    num_hidden_layers = 0
    hidden_dim = 21  # 7
    output_dim = 2
    inh_soma_dim = 1
    inh_dend_dim = 0
    seed = 0
    shuffle = True
    disp = True
    plot = 2
    max_weight = 2.
    min_inh_weight = -5.
    w_E_I_dend = -0.001
    w_E_I_soma = -0.2
    w_I_soma_E = 1.
    w_I_dend_E = 1.
    w_I_soma_E_scale = 2.
    w_I_soma_E_floor = 0.01
    I_soma_E_learning_rule = 'Anti-Hebb + weight norm'

    num_blocks = 300  # each block contains all input patterns

    output_layer_pos_mod_th = 0.25
    output_layer_neg_mod_th = -0.125
    hidden_layer_pos_mod_th = 0.125
    hidden_layer_neg_mod_th = -0.125

    output_layer_pos_mod_learning_rate = 0.25
    hidden_layer_pos_mod_learning_rate = 0.25

    E_bias_learning_rate = 0.25
    E_I_soma_learning_rate = 0.25
    I_soma_E_learning_rate = 0.25

    neg_mod_pre_discount = 0.25
    dep_ratio = 1.
    dep_th = 0.01
    dep_width = 0.01

    network = BTSP_network(num_hidden_layers=num_hidden_layers, input_dim=input_dim, hidden_dim=hidden_dim,
                           hidden_inh_soma_dim=inh_soma_dim, hidden_inh_dend_dim=inh_dend_dim,
                           output_dim=input_dim, output_inh_soma_dim=inh_soma_dim, tau=3, num_steps=10,
                           activation_f=ReLU, seed=seed)

    input_peak_rate = 1.
    input_pattern_matrix = np.array([[0., 0., 1., 1.],
                                     [0., 1., 0., 1.]])
    num_patterns = input_pattern_matrix.shape[1]

    target_output_pattern_matrix = np.array([[0., 1., 1., 0.],
                                             [0., 0., 0., 1.]])

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

    initial_FF_weight_bounds_dict = {}
    initial_FB_weight_bounds_dict = {}
    FF_weight_bounds_dict = {}
    FB_weight_bounds_dict = {}
    E_I_soma_weight_bounds = {}
    I_soma_E_weight_bounds = {}

    initial_E_I_dend_weight_bounds = {}
    initial_E_I_soma_weight_bounds = {}
    initial_I_soma_E_weight_bounds = {}
    initial_I_dend_E_weight_bounds = {}
    initial_E_bias_bounds = {}

    for layer in range(1, network.num_layers):
        curr_layer_dim = network.layer_dims[layer]
        prev_layer_dim = network.layer_dims[layer - 1]
        initial_FF_weight_bounds_dict[layer] = (0., 2. * max_weight / prev_layer_dim)
        FF_weight_bounds_dict[layer] = (0., max_weight)

        curr_layer_inh_soma_dim = network.inh_soma_layer_dims[layer]
        if curr_layer_inh_soma_dim > 1:
            raise Exception('inh_soma_dim > 1 not yet implemented')
        if curr_layer_inh_soma_dim > 0:
            initial_I_soma_E_weight_bounds[layer] = (w_I_soma_E, w_I_soma_E)
            initial_E_I_soma_weight_bounds[layer] = (w_E_I_soma, 0.)
            E_I_soma_weight_bounds[layer] = (min_inh_weight, 0.)
            I_soma_E_weight_bounds[layer] = (w_I_soma_E_floor, w_I_soma_E_scale)

    for layer in range(1, network.num_layers - 1):
        curr_layer_dim = network.layer_dims[layer]
        next_layer_dim = network.layer_dims[layer + 1]
        # initial_FB_weight_bounds_dict[layer] = (max_weight / next_layer_dim,
        #                                        1.5 * (max_weight / next_layer_dim))
        initial_FB_weight_bounds_dict[layer] = (0.75 * max_weight / next_layer_dim,
                                                1.25 * (max_weight / next_layer_dim))
        FB_weight_bounds_dict[layer] = (0., max_weight)
        curr_layer_inh_dend_dim = network.inh_dend_layer_dims[layer]
        if curr_layer_inh_dend_dim > 1:
            raise Exception('inh_dend_dim > 1 not yet implemented')
        if curr_layer_inh_dend_dim > 0:
            initial_E_I_dend_weight_bounds[layer] = (w_E_I_dend, w_E_I_dend)
            initial_I_dend_E_weight_bounds[layer] = (w_I_dend_E, w_I_dend_E)

    for layer in range(1, network.num_layers):
        initial_E_bias_bounds[layer] = (0., 0.)

    network.init_weights(FF_weight_bounds_dict, FB_weight_bounds_dict, E_I_soma_weight_bounds, I_soma_E_weight_bounds,
                         initial_FF_weight_bounds_dict, initial_FB_weight_bounds_dict, initial_I_soma_E_weight_bounds,
                         initial_I_dend_E_weight_bounds, initial_E_I_dend_weight_bounds, initial_E_I_soma_weight_bounds,
                         initial_E_bias_bounds)

    # network.initial_I_soma_E_weight_matrix_dict[1] = np.array([[0., 1.]])

    if plot > 1:
        fig, axes = plt.subplots(network.num_layers - 1, 3, figsize=(9, 3 * (network.num_layers - 1)))

        row = 0
        for layer in range(1, network.num_layers - 1):
            cbar = axes[row][0].imshow(network.initial_FF_weight_matrix_dict[layer], aspect='auto',
                                       interpolation='none')
            axes[row][0].set_title('Layer %i - FF weights' % (layer))
            axes[row][0].set_ylabel('Layer %i units' % (layer))
            axes[row][0].set_xlabel('Layer %i units' % (layer - 1))
            fig.colorbar(cbar, ax=axes[row][0])

            cbar = axes[row][1].imshow(network.initial_FB_weight_matrix_dict[layer], aspect='auto',
                                       interpolation='none')
            axes[row][1].set_title('Layer %i - FB weights' % (layer))
            axes[row][1].set_ylabel('Layer %i units' % (layer))
            axes[row][1].set_xlabel('Layer %i units' % (layer + 1))
            fig.colorbar(cbar, ax=axes[row][1])

            cbar = axes[row][2].imshow(np.atleast_2d(network.initial_E_I_dend_weight_matrix_dict[layer]),
                                       aspect='auto', interpolation='none')
            axes[row][2].set_title('Layer %i - E <- I_dend weights' % (layer))
            axes[row][2].set_ylabel('Layer %i units' % (layer))
            axes[row][2].set_xlabel('Layer %i inh units' % (layer))
            fig.colorbar(cbar, ax=axes[row][2])
            row += 1

        if network.num_hidden_layers > 0:
            layer += 1
            cbar = axes[row][0].imshow(network.initial_FF_weight_matrix_dict[layer], aspect='auto',
                                       interpolation='none')
            axes[row][0].set_title('Layer %i - FF weights' % (layer))
            axes[row][0].set_ylabel('Layer %i units' % (layer))
            axes[row][0].set_xlabel('Layer %i units' % (layer - 1))
            fig.colorbar(cbar, ax=axes[row][0])

        fig.tight_layout()
        fig.show()

    pos_mod_learning_rate_dict = {}
    pos_mod_th_dict = {}
    neg_mod_th_dict = {}
    for layer in range(1, network.num_layers):
        if layer != network.num_layers - 1:
            pos_mod_learning_rate_dict[layer] = hidden_layer_pos_mod_learning_rate
            pos_mod_th_dict[layer] = hidden_layer_pos_mod_th
            neg_mod_th_dict[layer] = hidden_layer_neg_mod_th
        else:
            pos_mod_learning_rate_dict[layer] = output_layer_pos_mod_learning_rate
            pos_mod_th_dict[layer] = output_layer_pos_mod_th
            neg_mod_th_dict[layer] = output_layer_neg_mod_th

    network.config_BTSP_rule(pos_mod_learning_rate_dict, pos_mod_th_dict, neg_mod_th_dict,
                             dep_ratio, dep_th, dep_width, neg_mod_pre_discount)

    E_bias_learning_rate_dict = {}
    for layer in range(1, network.num_layers):
        E_bias_learning_rate_dict[layer] = E_bias_learning_rate
    network.config_bias_learning_rule(E_bias_learning_rate_dict)

    E_I_soma_learning_rate_dict = {}
    for layer in range(1, network.num_layers):
        E_I_soma_learning_rate_dict[layer] = E_I_soma_learning_rate
    network.config_E_I_soma_learning_rule(E_I_soma_learning_rate_dict)

    I_soma_E_learning_rate_dict = {}
    I_soma_E_learning_rule_dict = {}
    I_soma_E_weight_scale_dict = {}
    for layer in range(1, network.num_layers):
        I_soma_E_learning_rate_dict[layer] = I_soma_E_learning_rate
        I_soma_E_learning_rule_dict[layer] = I_soma_E_learning_rule
        I_soma_E_weight_scale_dict[layer] = w_I_soma_E_scale
    network.config_I_soma_E_learning_rule(I_soma_E_learning_rate_dict, I_soma_E_learning_rule_dict,
                                          I_soma_E_weight_scale_dict)

    if plot > 1:
        BTSP_delta_w = network.pos_mod_rule_dict[network.num_layers - 1]
        pre = np.linspace(0., 2. * input_peak_rate, 100)
        w0 = np.linspace(0., max_weight, 100)
        pre_mesh, w0_mesh = np.meshgrid(pre, w0)
        fig = plt.figure()
        plt.pcolormesh(pre_mesh, w0_mesh, BTSP_delta_w(pre_mesh, w0_mesh, 1., max_weight), cmap='RdBu_r',
                       shading='nearest', vmin=-max_weight * network.pos_mod_learning_rate_dict[network.num_layers - 1],
                       vmax=max_weight * network.pos_mod_learning_rate_dict[network.num_layers - 1])
        plt.xlabel('Pre activity')
        plt.ylabel('Initial weight')
        cbar = plt.colorbar()
        cbar.set_label('Delta weight', rotation=-90.)
        fig.show()

    summed_FF_input_dict, summed_FB_input_dict, layer_activity_dict, layer_inh_soma_activity_dict, \
    layer_inh_dend_activity_dict = \
        network.get_layer_activities(network.input_pattern_matrix, network.initial_FF_weight_matrix_dict,
                                     network.initial_FB_weight_matrix_dict, network.initial_I_soma_E_weight_matrix_dict,
                                     network.initial_I_dend_E_weight_matrix_dict,
                                     network.initial_E_I_soma_weight_matrix_dict,
                                     network.initial_E_I_dend_weight_matrix_dict, network.initial_E_bias_array_dict)

    if plot > 0:
        network.plot_network_state_summary(network.initial_FF_weight_matrix_dict, network.initial_FB_weight_matrix_dict,
                                           summed_FF_input_dict, summed_FB_input_dict, layer_activity_dict,
                                           layer_inh_soma_activity_dict)

    context.update(locals())
    #plt.show()
    #return

    loss = network.train(num_blocks, shuffle=shuffle, disp=disp, plot=plot>0)

    summed_FF_input_dict, summed_FB_input_dict, layer_activity_dict, layer_inh_soma_activity_dict, \
    layer_inh_dend_activity_dict = \
            network.get_layer_activities(network.input_pattern_matrix, network.FF_weight_matrix_dict_history[-1],
                                         network.FB_weight_matrix_dict_history[-1],
                                         network.I_soma_E_weight_matrix_dict_history[-1],
                                         network.initial_I_dend_E_weight_matrix_dict,
                                         network.E_I_soma_weight_matrix_dict_history[-1],
                                         network.E_I_dend_weight_matrix_dict_history[-1],
                                         network.E_bias_array_dict_history[-1])
    if plot > 0:
        network.plot_network_state_summary(network.FF_weight_matrix_dict_history[-1],
                                           network.FB_weight_matrix_dict_history[-1], summed_FF_input_dict,
                                           summed_FB_input_dict, layer_activity_dict, layer_inh_soma_activity_dict)

    context.update(locals())

    plt.show()


if __name__ == '__main__':
    main()