from nested.optimize_utils import Context, param_array_to_dict
from train_Hebb_lateral_inh_network import *


context = Context()

def config_worker():

    ReLU = lambda x: np.maximum(0., x)

    input_dim = 21
    if 'num_hidden_layers' not in context():
        num_hidden_layers = 1
    else:
        num_hidden_layers = int(context.num_hidden_layers)
    hidden_dim = 7
    hidden_inh_dim = 7
    output_dim = 21
    output_inh_dim = 7
    tau = 3
    num_steps = 12
    seed = 0
    disp = True
    shuffle = True
    n_hot = 1
    I_floor_weight = -0.05
    if 'anti_Hebb_I' not in context():
        anti_Hebb_I = True
    if 'plot' not in context():
        plot = False
    if 'num_blocks' not in context():
        num_blocks = 400  # each block contains all input patterns
    else:
        num_blocks = int(context.num_blocks)

    network = Hebb_lat_inh_network(num_hidden_layers=num_hidden_layers, input_dim=input_dim, hidden_dim=hidden_dim,
                                   hidden_inh_dim=hidden_inh_dim, output_dim=output_dim, output_inh_dim=output_inh_dim,
                                   tau=tau, num_steps=num_steps, I_floor_weight=I_floor_weight, activation_f=ReLU,
                                   seed=seed)

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

    context.update(locals())


def compute_features(x, model_id, export=False):

    network = context.network
    loss = test_Hebb_lateral_inh_network(x, network, context.num_blocks, context.anti_Hebb_I, context.shuffle,
                                         context.param_names, disp=context.disp, plot=context.plot)

    if context.plot:
        layer_output_dict, layer_inh_output_dict = \
            network.get_layer_activities(network.input_pattern_matrix, network.E_E_weight_matrix_dict_history[-1],
                                         network.E_I_weight_matrix_dict_history[-1],
                                         network.I_E_weight_matrix_dict_history[-1],
                                         network.I_I_weight_matrix_dict_history[-1])

        network.plot_network_state_summary(network.E_E_weight_matrix_dict_history[-1],
                                           network.E_I_weight_matrix_dict_history[-1],
                                           network.I_E_weight_matrix_dict_history[-1],
                                           network.I_I_weight_matrix_dict_history[-1], layer_output_dict,
                                           layer_inh_output_dict)

    return {'MSE_loss': loss}


def get_objectives(features, model_id, export=False):

    return features, features


def test_Hebb_lateral_inh_network(x, network, num_blocks, anti_Hebb_I, shuffle, param_names, disp=False, plot=False):

    x_dict = param_array_to_dict(x, param_names)

    E_E_learning_rate = x_dict['E_E_learning_rate']
    E_I_learning_rate = x_dict['E_I_learning_rate']
    I_E_learning_rate = x_dict['I_E_learning_rate']
    I_I_learning_rate = x_dict['I_I_learning_rate']
    E_E_output_weight_scale = x_dict['E_E_output_weight_scale']
    E_I_output_weight_scale = x_dict['E_I_output_weight_scale']
    I_E_output_weight_scale = x_dict['I_E_output_weight_scale']
    I_I_output_weight_scale = x_dict['I_I_output_weight_scale']
    E_E_hidden_weight_scale = x_dict['E_E_hidden_weight_scale']
    E_I_hidden_weight_scale = x_dict['E_I_hidden_weight_scale']
    I_E_hidden_weight_scale = x_dict['I_E_hidden_weight_scale']
    I_I_hidden_weight_scale = x_dict['I_I_hidden_weight_scale']

    E_E_weight_scale_dict = {}
    E_I_weight_scale_dict = {}
    I_E_weight_scale_dict = {}
    I_I_weight_scale_dict = {}

    for layer in range(1, network.num_layers):
        if layer == network.num_layers - 1:
            E_E_weight_scale_dict[layer] = E_E_output_weight_scale
            if network.inh_layer_dims[layer] > 0:
                E_I_weight_scale_dict[layer] = E_I_output_weight_scale
                I_E_weight_scale_dict[layer] = I_E_output_weight_scale
                I_I_weight_scale_dict[layer] = I_I_output_weight_scale
        else:
            E_E_weight_scale_dict[layer] = E_E_hidden_weight_scale
            if network.inh_layer_dims[layer] > 0:
                E_I_weight_scale_dict[layer] = E_I_hidden_weight_scale
                I_E_weight_scale_dict[layer] = I_E_hidden_weight_scale
                I_I_weight_scale_dict[layer] = I_I_hidden_weight_scale

    network.init_weights(E_E_weight_scale_dict, E_I_weight_scale_dict, I_E_weight_scale_dict, I_I_weight_scale_dict)

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
            I_E_learning_rule_dict[layer] = 'Hebb + weight norm'
            if anti_Hebb_I:
                E_I_learning_rule_dict[layer] = 'Anti-Hebb + weight norm'
                I_I_learning_rule_dict[layer] = 'Anti-Hebb + weight norm'
            else:
                E_I_learning_rule_dict[layer] = 'Hebb + weight norm'
                I_I_learning_rule_dict[layer] = 'Hebb + weight norm'

    network.config_learning_rules(E_E_learning_rule_dict, E_I_learning_rule_dict, I_E_learning_rule_dict,
                                  I_I_learning_rule_dict, E_E_learning_rate_dict, E_I_learning_rate_dict,
                                  I_E_learning_rate_dict, I_I_learning_rate_dict)

    if disp:
        print('x: %s' % str(list(x)))
    loss = network.train(num_blocks, shuffle=shuffle, disp=disp, plot=plot)

    return loss

