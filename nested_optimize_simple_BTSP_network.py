from nested.optimize_utils import Context, param_array_to_dict
from train_simple_BTSP_network import *


context = Context()

def config_worker():

    ReLU = lambda x: np.maximum(0., x)

    n_hot = 1
    input_dim = 21
    num_hidden_layers = 1
    hidden_dim = 21  # 7
    output_dim = 21
    inh_soma_dim = 1
    inh_dend_dim = 1
    seed = 0
    shuffle = True
    disp = True
    plot = 2
    
    dep_ratio = 1.
    dep_th = 0.01
    dep_width = 0.01

    num_blocks = 100  # each block contains all input patterns

    network = BTSP_network(num_hidden_layers=num_hidden_layers, input_dim=input_dim, hidden_dim=hidden_dim,
                           hidden_inh_soma_dim=inh_soma_dim, hidden_inh_dend_dim=inh_dend_dim,
                           output_dim=input_dim, output_inh_soma_dim=inh_soma_dim, tau=3, num_steps=12,
                           activation_f=ReLU, seed=seed)

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
                                    context.dep_width, context.shuffle, context.param_names, disp=context.disp,
                                    plot=context.plot)

    if context.plot:
        summed_FF_input_dict, summed_FB_input_dict, layer_activity_dict, layer_inh_soma_activity_dict, \
        layer_inh_dend_activity_dict = \
            network.get_layer_activities(network.input_pattern_matrix, network.FF_weight_matrix_dict_history[-1],
                                         network.FB_weight_matrix_dict_history[-1],
                                         network.initial_I_soma_E_weight_matrix_dict,
                                         network.initial_I_dend_E_weight_matrix_dict,
                                         network.initial_E_I_soma_weight_matrix_dict,
                                         network.E_I_dend_weight_matrix_dict_history[-1])
        network.plot_network_state_summary(network.FF_weight_matrix_dict_history[-1],
                                           network.FB_weight_matrix_dict_history[-1], summed_FF_input_dict,
                                           summed_FB_input_dict, layer_activity_dict, layer_inh_soma_activity_dict)

    return {'MSE_loss': loss}


def get_objectives(features, model_id, export=False):

    return features, features


def test_simple_BTSP_network(x, network, num_blocks, dep_ratio, dep_th, dep_width, shuffle, param_names, disp=False,
                             plot=False):
    """

    :param x:
    :param network:
    :param num_blocks:
    :param dep_ratio:
    :param dep_th:
    :param dep_width:
    :param shuffle:
    :param param_names:
    :param disp:
    :param plot:
    :return:
    """

    x_dict = param_array_to_dict(x, param_names)

    hidden_FF_max_weight = x_dict['hidden_FF_max_weight']
    hidden_FB_max_weight = x_dict['hidden_FB_max_weight']
    output_FF_max_weight = x_dict['output_FF_max_weight']
    output_layer_pos_mod_th = x_dict['output_layer_pos_mod_th']
    output_layer_neg_mod_th = x_dict['output_layer_neg_mod_th']
    hidden_layer_pos_mod_th = x_dict['hidden_layer_pos_mod_th']
    hidden_layer_neg_mod_th = x_dict['hidden_layer_neg_mod_th']
    output_layer_pos_mod_learning_rate = x_dict['output_layer_pos_mod_learning_rate']
    hidden_layer_pos_mod_learning_rate = x_dict['hidden_layer_pos_mod_learning_rate']
    FF_max_init_weight_factor = x_dict['FF_max_init_weight_factor']
    FB_min_init_weight_factor = x_dict['FB_min_init_weight_factor']
    FB_max_init_weight_factor = x_dict['FB_max_init_weight_factor']
    E_I_soma_max_init_weight = x_dict['E_I_soma_max_init_weight']
    I_soma_E_max_init_weight = x_dict['I_soma_E_max_init_weight']
    E_I_dend_max_init_weight = x_dict['E_I_dend_max_init_weight']
    I_dend_E_max_init_weight = x_dict['I_dend_E_max_init_weight']
    neg_mod_pre_discount = x_dict['neg_mod_pre_discount']
    
    initial_FF_weight_bounds_dict = {}
    initial_FB_weight_bounds_dict = {}
    FF_weight_bounds_dict = {}
    FB_weight_bounds_dict = {}
    initial_E_I_dend_weight_bounds = {}
    initial_E_I_soma_weight_bounds = {}
    initial_I_soma_E_weight_bounds = {}
    initial_I_dend_E_weight_bounds = {}

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

        curr_layer_inh_soma_dim = network.inh_soma_layer_dims[layer]
        if curr_layer_inh_soma_dim > 0:
            if curr_layer_inh_soma_dim > 1:
                raise Exception('inh_soma_dim > 1 not yet implemented')
            else:
                initial_I_soma_E_weight_bounds[layer] = (I_soma_E_max_init_weight, I_soma_E_max_init_weight)
                initial_E_I_soma_weight_bounds[layer] = (E_I_soma_max_init_weight, E_I_soma_max_init_weight)
    
    for layer in range(1, network.num_layers - 1):
        curr_layer_dim = network.layer_dims[layer]
        next_layer_dim = network.layer_dims[layer + 1]
        initial_FB_weight_bounds_dict[layer] = (FB_min_init_weight_factor * hidden_FB_max_weight / next_layer_dim,
                                                FB_max_init_weight_factor * hidden_FB_max_weight / next_layer_dim)
        FB_weight_bounds_dict[layer] = (0., hidden_FB_max_weight)
        
        curr_layer_inh_dend_dim = network.inh_dend_layer_dims[layer]
        if curr_layer_inh_dend_dim > 0:
            if curr_layer_inh_dend_dim > 1:
                raise Exception('inh_dend_dim > 1 not yet implemented')
            else:
                initial_I_dend_E_weight_bounds[layer] = (I_dend_E_max_init_weight, I_dend_E_max_init_weight)
                initial_E_I_dend_weight_bounds[layer] = (E_I_dend_max_init_weight, E_I_dend_max_init_weight)
    
    network.init_weights(FF_weight_bounds_dict, FB_weight_bounds_dict, initial_FF_weight_bounds_dict,
                         initial_FB_weight_bounds_dict, initial_I_soma_E_weight_bounds, initial_I_dend_E_weight_bounds,
                         initial_E_I_dend_weight_bounds, initial_E_I_soma_weight_bounds)

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

    if disp:
        print('x: %s' % str(list(x)))
    loss = network.train(num_blocks, shuffle=shuffle, disp=disp, plot=plot)

    return loss

