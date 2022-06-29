from nested.optimize_utils import Context, param_array_to_dict
from train_simple_BTSP_network import *


context = Context()


def config_worker():

    ReLU = lambda x: np.maximum(0., x)

    input_dim = 21
    num_hidden_layers = 1
    hidden_dim = 7
    inh_dim = 1
    seed = 0

    w_inh_hidden = 1.
    dep_ratio = 1.
    dep_th = 0.01
    dep_width = 0.01

    num_blocks = 20  # each block contains all input patterns

    network = BTSP_network(num_hidden_layers=num_hidden_layers, input_dim=input_dim, hidden_dim=hidden_dim,
                           inh_dim=inh_dim, output_dim=input_dim, activation_f=ReLU, seed=seed)

    input_peak_rate = 1.
    input_pattern_matrix = np.zeros([input_dim,input_dim])
    input_indexes = np.arange(0, input_dim)
    for i in input_indexes:
        input_pattern_matrix[i,i] = input_peak_rate * 1.
    target_output_pattern_matrix = np.copy(input_pattern_matrix)

    network.load_input_patterns(input_pattern_matrix)
    network.load_target_output_patterns(target_output_pattern_matrix)

    context.update(locals())


def compute_features(x, model_id, export=False):

    network = context.network
    loss = test_simple_BTSP_network(x, network, context.num_blocks, context.dep_ratio, context.dep_th,
                                    context.dep_width, context.w_inh_hidden, context.param_names, disp=context.disp,
                                    plot=context.plot)

    if context.plot:
        summed_FF_input_dict, summed_FB_input_dict, layer_output_dict, layer_inh_output_dict = \
            context.network.get_layer_activities(network.input_pattern_matrix,
                                                 network.FF_weight_matrix_dict_history[-1],
                                                 network.FB_weight_matrix_dict_history[-1],
                                                 network.initial_inh_hidden_weight_matrix_dict,
                                                 network.initial_hidden_inh_weight_matrix_dict)
        network.plot_network_state_summary(network.FF_weight_matrix_dict_history[-1],
                                           network.FB_weight_matrix_dict_history[-1], summed_FF_input_dict,
                                           summed_FB_input_dict, layer_output_dict, layer_inh_output_dict)

    return {'MSE_loss': loss}


def get_objectives(features, model_id, export=False):

    return features, features


def test_simple_BTSP_network(x, network, num_blocks, dep_ratio, dep_th, dep_width, w_inh_hidden, param_names,
                             disp=False, plot=False):

    x_dict = param_array_to_dict(x, param_names)

    hidden_FF_max_weight = x_dict['hidden_FF_max_weight']
    hidden_FB_max_weight = x_dict['hidden_FB_max_weight']
    output_FF_max_weight = x_dict['output_FF_max_weight']
    w_hidden_inh = x_dict['w_hidden_inh']
    output_layer_pos_mod_th = x_dict['output_layer_pos_mod_th']
    output_layer_neg_mod_th = x_dict['output_layer_neg_mod_th']
    hidden_layer_pos_mod_th = x_dict['hidden_layer_pos_mod_th']
    hidden_layer_neg_mod_th = x_dict['hidden_layer_neg_mod_th']
    output_layer_pos_mod_learning_rate = x_dict['output_layer_pos_mod_learning_rate']
    hidden_layer_pos_mod_learning_rate = x_dict['hidden_layer_pos_mod_learning_rate']
    output_layer_neg_mod_learning_rate = x_dict['output_layer_neg_mod_learning_rate']
    hidden_layer_neg_mod_learning_rate = x_dict['hidden_layer_neg_mod_learning_rate']
    FF_max_init_weight_factor = x_dict['FF_max_init_weight_factor']
    FB_min_init_weight_factor = x_dict['FB_min_init_weight_factor']
    FB_max_init_weight_factor = x_dict['FB_max_init_weight_factor']

    initial_FF_weight_bounds_dict = {}
    initial_FB_weight_bounds_dict = {}
    FF_weight_bounds_dict = {}
    FB_weight_bounds_dict = {}
    initial_hidden_inh_weight_bounds = {}
    initial_inh_hidden_weight_bounds = {}

    for layer in range(1, network.num_layers):
        curr_layer_dim = network.layer_dims[layer]
        prev_layer_dim = network.layer_dims[layer - 1]
        if layer == 1:
            initial_FF_weight_bounds_dict[layer] = \
                (0., FF_max_init_weight_factor * hidden_FF_max_weight / prev_layer_dim)
            FF_weight_bounds_dict[layer] = (0., hidden_FF_max_weight)
        else:
            initial_FF_weight_bounds_dict[layer] = \
                (0., FF_max_init_weight_factor * output_FF_max_weight / prev_layer_dim)
            FF_weight_bounds_dict[layer] = (0., output_FF_max_weight)

    for layer in range(1, network.num_layers - 1):
        curr_layer_dim = network.layer_dims[layer]
        next_layer_dim = network.layer_dims[layer + 1]
        initial_FB_weight_bounds_dict[layer] = (FB_min_init_weight_factor * hidden_FB_max_weight / next_layer_dim,
                                                FB_max_init_weight_factor * hidden_FB_max_weight / next_layer_dim)
        FB_weight_bounds_dict[layer] = (0., hidden_FB_max_weight)
        if network.inh_dim > 1:
            raise Exception('inh_dim > 1 not yet implemented')
        initial_hidden_inh_weight_bounds[layer] = (w_hidden_inh, w_hidden_inh)
        initial_inh_hidden_weight_bounds[layer] = (w_inh_hidden, w_inh_hidden)

    network.init_weights(FF_weight_bounds_dict, FB_weight_bounds_dict, initial_FF_weight_bounds_dict,
                         initial_FB_weight_bounds_dict, initial_inh_hidden_weight_bounds,
                         initial_hidden_inh_weight_bounds)

    pos_mod_learning_rate_dict = {}
    neg_mod_learning_rate_dict = {}
    pos_mod_th_dict = {}
    neg_mod_th_dict = {}
    for layer in range(1, network.num_layers):
        if layer != network.num_layers - 1:
            pos_mod_learning_rate_dict[layer] = hidden_layer_pos_mod_learning_rate
            neg_mod_learning_rate_dict[layer] = hidden_layer_neg_mod_learning_rate
            pos_mod_th_dict[layer] = hidden_layer_pos_mod_th
            neg_mod_th_dict[layer] = hidden_layer_neg_mod_th
        else:
            pos_mod_learning_rate_dict[layer] = output_layer_pos_mod_learning_rate
            neg_mod_learning_rate_dict[layer] = output_layer_neg_mod_learning_rate
            pos_mod_th_dict[layer] = output_layer_pos_mod_th
            neg_mod_th_dict[layer] = output_layer_neg_mod_th

    network.config_BTSP_rule(pos_mod_learning_rate_dict, neg_mod_learning_rate_dict, pos_mod_th_dict, neg_mod_th_dict,
                             dep_ratio, dep_th, dep_width)
    if disp:
        print('x: %s' % str(list(x)))
    loss = network.train(num_blocks, disp=disp, plot=plot)

    return loss

