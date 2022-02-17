from spiking_BTSP_network_utils import *

dt = 0.001
t_lim = 15.

input_dim = 10
hidden_dim = 10
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

for i, loc in enumerate(input_locs):
    indexes = np.where((network.t >= loc + teaching_window) & (network.t < loc + teaching_window + teaching_input_dur))
    teaching_input[i, indexes] = 1.

plt.figure()
plt.imshow(teaching_input, aspect='auto', cmap='binary', interpolation='none')
plt.ylabel('Output unit ID')
plt.xlabel('Time (sec)')
plt.title('Teaching inputs to output layer dendrite')
plt.show()


input_layer = InputLayer(network, input_dim)
network.init_input_layer(input_layer, patterns=input_pattern)
hidden1 = HiddenDendLayer(network, hidden_dim)
network.append_hidden_layer(hidden1)
output_layer = OutputDendLayer(network, output_dim, target_eligibility=target_eligibility)
network.init_output_layer(output_layer, teaching_input)

BTSP_rule = BTSPLearningRule()
BTSP_rule.plot()
plt.show()

network.layers['Hidden1'].connect('Input', 'soma', BTSP_rule)
network.layers['Hidden1'].connect('Output', 'dend', BTSP_rule)
network.output_layer.connect('Hidden1', 'soma', BTSP_rule)

local_random = np.random.default_rng(seed=0)
network.layers['Hidden1'].weights['Input'] = local_random.uniform(0., 0., (hidden_dim, input_dim))
network.layers['Hidden1'].weights['Output'] = local_random.uniform(0., 0.2, (hidden_dim, output_dim))
network.output_layer.weights['Hidden1'] = local_random.uniform(0., 0., (output_dim, hidden_dim))

pattern_index = 0
num_epochs = 2
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

plt.show()

