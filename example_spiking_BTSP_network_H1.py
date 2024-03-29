from spiking_BTSP_network_utils import *
plot_level = 1

dt = 0.001
t_lim = 15.

input_dim = 10
hidden_dim = 10
inh_dim = 1
output_dim = 10

network = Network(dt, t_lim)

input_rate = 25.  # Hz
input_dur = 1.  # sec
input_locs = np.arange(0., input_dim, 1.)
input_pattern = np.zeros((input_dim, len(network.t)))
for i in range(input_dim):
    dur = 0.
    ISI = 1. / input_rate
    while dur < input_dur:
        index = np.where(network.t >= input_locs[i] + dur)[0][0]
        input_pattern[i, index] = 1.
        dur += ISI

if plot_level > 1:
    plt.figure()
    indexes = np.where(input_pattern > 0.)
    plt.scatter(network.t[indexes[1]], indexes[0], marker='.', s=0.5, c='k')
    plt.ylim(input_dim-0.5, -0.5)
    plt.ylabel('Input unit ID')
    plt.xlabel('Time (sec)')
    plt.title('Spiking activity of input units')
    plt.show()


target_eligibility = 0.5
teaching_input = np.zeros((output_dim, len(network.t)))
teaching_window = 0.5  # sec
teaching_input_dur = 0.3  # sec
dend_v_ca_spike_th = 0.6
soma_v_spike_th = 0.3
dend_soma_coupling = 0.05
inh_learning_rate = 0.4
inh_soma_v_spike_th = 0.1
inh_soma_v_act_rate = 20.
inh_soma_v_decay_tau = 0.1

for i, loc in enumerate(input_locs):
    indexes = np.where((network.t >= loc + teaching_window) & (network.t < loc + teaching_window + teaching_input_dur))
    teaching_input[i, indexes] = 1.

if plot_level > 1:
    plt.figure()
    plt.imshow(teaching_input, aspect='auto', cmap='binary', interpolation='none')
    plt.ylabel('Output unit ID')
    plt.xlabel('Time (sec)')
    plt.title('Teaching inputs to output layer dendrite')
    plt.show()


input_layer = InputLayer(network, input_dim)
network.init_input_layer(input_layer, patterns=input_pattern)
hidden1 = HiddenDendLayer(network, hidden_dim, dend_v_ca_spike_th=dend_v_ca_spike_th,
                          dend_soma_coupling=dend_soma_coupling, soma_v_spike_th=soma_v_spike_th)
network.append_hidden_layer(hidden1)
inh1 = SomaLayer(network, inh_dim, label='Inh1', soma_v_spike_th=inh_soma_v_spike_th,
                 soma_v_act_rate=inh_soma_v_act_rate, soma_v_decay_tau=inh_soma_v_decay_tau)
network.append_hidden_layer(inh1)
output_layer = OutputDendLayer(network, output_dim, dend_v_ca_spike_th=dend_v_ca_spike_th,
                               target_eligibility=target_eligibility, soma_v_spike_th=soma_v_spike_th,
                               dend_soma_coupling=dend_soma_coupling)
network.init_output_layer(output_layer, teaching_input)
inh_out = SomaLayer(network, inh_dim, label='Inh_Out', soma_v_spike_th=inh_soma_v_spike_th,
                    soma_v_act_rate=inh_soma_v_act_rate, soma_v_decay_tau=inh_soma_v_decay_tau)
network.append_hidden_layer(inh_out)

BTSP_rule = BTSPLearningRule()
if plot_level > 1:
    BTSP_rule.plot()

DendInhHomeo_rule = DendInhHomeoLearningRule(learning_rate=inh_learning_rate, th=4.*dend_v_ca_spike_th)

network.layers['Hidden1'].connect('Input', 'soma', BTSP_rule)
network.layers['Hidden1'].connect('Output', 'dend', BTSP_rule)
network.layers['Hidden1'].connect('Inh1', 'dend', DendInhHomeo_rule)
network.layers['Inh1'].connect('Input', 'soma')
network.layers['Inh1'].connect('Hidden1', 'soma')
network.output_layer.connect('Hidden1', 'soma', BTSP_rule)
network.layers['Inh_Out'].connect('Output', 'soma')
network.output_layer.connect('Inh_Out', 'dend', DendInhHomeo_rule)

local_random = np.random.default_rng(seed=0)
network.layers['Hidden1'].weights['Input'] = local_random.uniform(0., 0.2, (hidden_dim, input_dim))
network.layers['Hidden1'].weights['Output'] = local_random.normal(1.0, 0.2, (hidden_dim, output_dim))
network.layers['Hidden1'].weights['Inh1'] = np.ones((hidden_dim, inh_dim)) * -3.
network.layers['Inh1'].weights['Input'] = np.ones((inh_dim, hidden_dim)) * 0.1
network.layers['Inh1'].weights['Hidden1'] = np.ones((inh_dim, hidden_dim)) * 0.1
network.layers['Inh_Out'].weights['Output'] = np.ones((inh_dim, output_dim)) * 1.
network.layers['Output'].weights['Inh_Out'] = np.ones((hidden_dim, inh_dim)) * -3.
network.output_layer.weights['Hidden1'] = local_random.uniform(0., 0.2, (output_dim, hidden_dim))


pattern_index = 0

num_epochs = 1
for epoch in range(1, 1 + num_epochs):
    network.train_pattern(pattern_index)

    fig = plt.figure()
    indexes = np.where(network.output_layer.activity > 0.)
    plt.scatter(network.t[indexes[1]], indexes[0], marker='.', s=0.5, c='k')
    plt.ylim(input_dim - 0.5, -0.5)
    plt.xlabel('Time (sec)')
    plt.ylabel('Output unit ID')
    plt.title('Output layer spiking activity during training; pattern %i, epoch %i' % (pattern_index, epoch))
    fig.show()

    fig = plt.figure()
    im = plt.imshow(network.output_layer.dend_ca_activity, aspect='auto', interpolation='none', cmap='binary',
                    extent=(0., network.t_lim, network.output_layer.dim, 0))
    plt.xlabel('Time (sec)')
    plt.ylabel('Output unit ID')
    plt.title('Output layer dend Ca spike activity; pattern %i, epoch %i' % (pattern_index, epoch))
    fig.show()

    fig = plt.figure()
    indexes = np.where(network.layers['Hidden1'].activity > 0.)
    plt.scatter(network.t[indexes[1]], indexes[0], marker='.', s=0.5, c='k')
    plt.ylim(input_dim - 0.5, -0.5)
    plt.xlabel('Time (sec)')
    plt.ylabel('Output unit ID')
    plt.title('Hidden layer spiking activity during training; pattern %i, epoch %i' % (pattern_index, epoch))
    fig.show()

    fig = plt.figure()
    im = plt.imshow(network.layers['Hidden1'].dend_ca_activity, aspect='auto', interpolation='none', cmap='binary',
                    extent=(0., network.t_lim, network.output_layer.dim, 0))
    plt.xlabel('Time (sec)')
    plt.ylabel('Output unit ID')
    plt.title('Hidden layer dend Ca spike activity; pattern %i, epoch %i' % (pattern_index, epoch))
    fig.show()

"""
network.show_pattern(pattern_index)

fig = plt.figure()
indexes = np.where(network.layers['Hidden1'].activity > 0.)
plt.scatter(network.t[indexes[1]], indexes[0], marker='.', s=0.5, c='k')
plt.ylim(input_dim-0.5, -0.5)
plt.xlabel('Time (sec)')
plt.ylabel('Output unit ID')
plt.title('Hidden layer spiking activity after training')
fig.show()

fig = plt.figure()
indexes = np.where(network.output_layer.activity > 0.)
plt.scatter(network.t[indexes[1]], indexes[0], marker='.', s=0.5, c='k')
plt.ylim(input_dim-0.5, -0.5)
plt.xlabel('Time (sec)')
plt.ylabel('Output unit ID')
plt.title('Output layer spiking activity after training')
fig.show()
# """

if plot_level > 0:
    plt.show()

