import numpy as np
import matplotlib.pyplot as plt
import scipy.signal


def get_osc(t, freq, depth, t_offset):
    return depth * (np.cos(2. * np.pi * freq * (t - t_offset) / 1000.) + 1.) / 2. + (1 - depth)


def get_gauss(x, peak_loc, width, depth=1, wrap=False):
    if wrap:
        dx = x[1] - x[0]
        length = dx * len(x)
        extended_x = np.concatenate([x - length, x, x + length])
    else:
        extended_x = x
    sigma = width / 3. / np.sqrt(2.)
    extended_rate = np.exp(-((extended_x - peak_loc) / sigma) ** 2)
    if wrap:
        rate = np.maximum(extended_rate[:len(x)], extended_rate[len(x):-len(x)])
        rate = np.maximum(rate, extended_rate[-len(x):])
    else:
        rate = extended_rate
    rate = (1. - depth) + depth * rate
    return rate


def get_spatial_osc_population_rate(t, freq, osc_depth, t_peak_locs, global_t_offset, t_spatial_width, spatial_depth,
                                    compression, peak_rate, wrap=False):
    """

    :param t: float (ms)
    :param freq: float (Hz)
    :param osc_depth: float in [0, 1]
    :param t_peak_locs: float (ms)
    :param global_t_offset: float (ms)
    :param t_spatial_width: float (s)
    :param spatial_depth: float in [0, 1]
    :param peak_rate: float (Hz)
    :param wrap: bool
    :return: 2d array of float
    """
    rate_matrix = np.empty((len(t_peak_locs), len(t)))
    for i, t_peak in enumerate(t_peak_locs):
        if compression is None:
            t_offset = global_t_offset
        else:
            t_offset = compression * t_peak + global_t_offset
        osc_rate = get_osc(t, freq, osc_depth, t_offset)
        spatial_rate = get_gauss(t, t_peak, t_spatial_width, spatial_depth, wrap)
        rate_matrix[i] = osc_rate * spatial_rate * peak_rate

    return rate_matrix


def get_d_cell_voltage_dt(cell_voltage, net_current, cell_tau, input_resistance=1.):
    """
    Computes the rate of change of cellular voltage for one postsynaptic unit.
    :param cell_voltage: float (mV)
    :param net_current: float (nA)
    :param cell_tau: float (seconds)
    :param input_resistance: float
    :return: float
    """
    d_cell_voltage_dt = (-cell_voltage + input_resistance * net_current) / cell_tau
    return d_cell_voltage_dt


def get_d_conductance_dt_array(channel_conductance, pre_activity, rise_tau, decay_tau):
    """

    :param channel_conductance: 1d array of float
    :param pre_activity: 1d array of float
    :param rise_tau: float (ms)
    :param decay_tau: float (ms)
    :return: 1d array of float
    """
    d_conductance_dt_array = -channel_conductance / decay_tau + \
                             np.maximum(0., pre_activity[:, None] - channel_conductance) / rise_tau
    return d_conductance_dt_array


def get_net_current(weights, channel_conductances, cell_voltage, reversal_potential):
    """

    :param weights: array of float
    :param channel_conductances: array of float
    :param cell_voltage: float (mV)
    :param reversal_potential: float (mV)
    :return:
    """
    net_current_array = ((weights * channel_conductances) * (reversal_potential - cell_voltage))
    net_current_array = np.sum(net_current_array, axis=0)
    return net_current_array


def get_d_network_intermediates_dt_dicts(num_units_dict, synapse_tau_dict, cell_tau_dict, weight_dict,
                                         weight_config_dict, synaptic_reversal_dict, channel_conductance_dict,
                                         cell_voltage_dict, network_activity_dict):
    """
    Computes rates of change of all synaptic currents and all cell voltages for all populations in a network.
    :param num_units_dict: dict: {'population': int (number of units in this population)}
    :param synapse_tau_dict: nested dict:
        {'postsynaptic population label':
            {'presynaptic population label':
                float (seconds)
            }
        }
    :param cell_tau_dict:
        {'population label':
            float (seconds)
        }
    :param weight_dict: nested dict:
        {'postsynaptic population label':
            {'presynaptic population label':
                2d array of float (number of presynaptic units, number of postsynaptic units)
            }
        }
    :param weight_config_dict: nested dict:
        {'postsynaptic population label':
            {'resynaptic population label':
                {'distribution: string
    :param syn_current_dict:
        {'post population label':
            {'pre population label':
                2d array of float (number of presynaptic units, number of postsynaptic units)
            }
        }
    :param cell_voltage_dict:
        {'population label':
            1d array of float (number of units)
        }
    :param network_activity_dict:
        {'population label':
            1d array of float (number of units)
        }
    :return: tuple of dict: (d_syn_current_dt_dict, d_cell_voltage_dt_dict)
    """
    d_syn_current_dt_dict = {}
    d_cell_voltage_dt_dict = {}
    d_conductance_dt_dict = {}

    for post_population in weight_dict:  # get the change in synaptic currents for every connection
        d_conductance_dt_dict[post_population] = {}
        this_cell_tau = cell_tau_dict[post_population]
        this_net_current = np.zeros_like(network_activity_dict[post_population])
        this_cell_voltage = cell_voltage_dict[post_population]
        for pre_population in weight_dict[post_population]:

            this_decay_tau = synapse_tau_dict[post_population][pre_population]['decay']
            this_rise_tau = synapse_tau_dict[post_population][pre_population]['rise']

            this_channel_conductance = channel_conductance_dict[post_population][pre_population]

            this_pre_activity = network_activity_dict[pre_population]
            d_conductance_dt_dict[post_population][pre_population] = \
                get_d_conductance_dt_array(this_channel_conductance, this_pre_activity, this_rise_tau, this_decay_tau)

            this_weights = weight_dict[post_population][pre_population]

            this_connection_type = weight_config_dict[post_population][pre_population]['connection_type']
            this_reversal_potential = synaptic_reversal_dict[this_connection_type]

            # TODO(done): np.sum should be inside get_net_current
            this_net_current += get_net_current(this_weights, channel_conductance_dict[post_population][pre_population],
                                                       this_cell_voltage, this_reversal_potential)
            #this_net_current += np.sum(syn_current_dict[post_population][pre_population], axis=0)
        d_cell_voltage_dt_dict[post_population] = \
            get_d_cell_voltage_dt_array(this_cell_voltage, this_net_current, this_cell_tau)

    return d_conductance_dt_dict, d_cell_voltage_dt_dict


duration = 10000. # ms
dt = 1  # ms
velocity = 20.  # cm / s
t = np.arange(0., duration, dt)
position = t / 1000. * velocity

global_t_offset = 0.  # ms
LFP_freq = 7 # Hz
input_freq = {'Inh': LFP_freq}
compression = {'Inh': None}

# Geisler et al., PNAS, 2010.
# cm
x_field_width = {'CA3': 60.,
                 'Inh': 60.}

t_field_width = {}
for pop in x_field_width:
    t_field_width[pop] = x_field_width[pop] / velocity * 1000.  # ms

input_freq['CA3'] = LFP_freq + 1. / t_field_width['CA3'] * 1000.
compression['CA3'] = 1. / (t_field_width['CA3'] / 1000. * input_freq['CA3'])

osc_depth = {'CA3': 0.7,
             'Inh': 0.5}

spatial_depth = {'CA3': 1.,
                 'Inh': 0.}

pop_size = {'CA3': 200,
            'Inh': 60}

# ms
syn_rise_tau = {'AMPA': 1., 'NMDA': 1., 'GABA': 1.}
syn_decay_tau = {'AMPA': 5., 'NMDA': 50., 'GABA': 5.}

# Hz
peak_rate = {'CA3': 40.,
             'Inh': 20.}

# mV
vm_rev = {'AMPA': 0.,
          'NMDA': 0.,
          'GABA': -70.}

vm_tau = 20.  # ms
vm_rest = -68.  # mV

t_peak_locs = {}
for pop in pop_size:
    t_peak_locs[pop] = np.linspace(-duration/2., duration + duration / 2., pop_size[pop])

input_rate = {}
for pop in input_freq:
    input_rate[pop] = get_spatial_osc_population_rate(t, input_freq[pop], osc_depth[pop], t_peak_locs[pop],
                                                      global_t_offset, t_field_width[pop], spatial_depth[pop],
                                                      compression[pop], peak_rate[pop])