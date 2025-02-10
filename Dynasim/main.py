import matplotlib.pyplot as plt
import numpy as np
import torch

from Neuronal.dlNetworks import *
from Neuronal.Dynalearn import *
from Neuronal.dlTools import *


# Example usage
dt = 0.1
num_neurons = [70, 15, 15]  # Number of neurons in each population
population_types = ["EX", "PV", "SST"]  # Population types
synGain = 2.0

connections = [
    (0, 0, 'AMPA', synGain * (torch.rand(num_neurons[0], num_neurons[0]) > 0.9)),  # EX to EX
    (0, 1, 'AMPA', synGain * (torch.rand(num_neurons[1], num_neurons[0]) > 0.9)),  # EX to SST
    (0, 2, 'NMDA', synGain * (torch.rand(num_neurons[2], num_neurons[0]) > 0.9)),  # EX to PV
    (1, 0, 'GABAb', synGain * (torch.rand(num_neurons[0], num_neurons[1]) > 0.9)),  # SST to EX
    (1, 1, 'GABAb', synGain * (torch.rand(num_neurons[1], num_neurons[1]) > 0.9)),  # SST to SST
    (1, 2, 'GABAb', synGain * (torch.rand(num_neurons[2], num_neurons[1]) > 0.9)),  # SST to PV
    (2, 0, 'GABAa', synGain * (torch.rand(num_neurons[0], num_neurons[2]) > 0.9)),    # PV to EX
    (2, 1, 'GABAa', synGain * (torch.rand(num_neurons[1], num_neurons[2]) > 0.9)),    # PV to SST
    (2, 2, 'GABAa', synGain * (torch.rand(num_neurons[2], num_neurons[2]) > 0.9))     # PV to PV
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
print(device)
network = HHNeurons(num_neurons, population_types, connections, dt=dt, inoise=5, device=device).to(device)

# Define external input currents
I_ext = (torch.rand(sum(num_neurons)) > 0.5) * 1.0  # External input to all populations
I_ext = I_ext.to(device)

V_histories = torch.zeros([10000, 100])

for i in range(10):
    igain = np.min([i, 5])
    V_histories[i*1000:(i+1)*1000, :] = network(I_ext*igain, T=100)

print(V_histories.shape)
# dlraster(V_histories, threshold=-20, dt=dt)
dlpotential(V_histories, dt=dt)
dlSpectrogram(V_histories)
dlConnections(network)

target_firing_rate = torch.ones([1])*5.0  # Example target firing rates
target_firing_rate = target_firing_rate.to(device)
# dynalearn = Dynalearn(network, lr=0.00001)
# dynalearn.optimize(I_ext, target_firing_rate, num_epochs=2)

#
# # Simulate network activity
# V_histories = network(I_ext, T=1000)
# print(V_histories.shape)
# dlraster(V_histories, threshold=-20, dt=dt)
# dlpotential(V_histories, dt=dt)
# dlConnections(network)

# V_histories = network(I_ext*2, T=1000)
# print(V_histories.shape)
# dlraster(V_histories, threshold=-20, dt=dt)
# dlpotential(V_histories, dt=dt)
# plt.imshow(network.synaptic_weights.detach().numpy())
# plt.show()
#
#
# V_histories = network(I_ext*2, T=100)
# print(V_histories.shape)
# dlraster(V_histories, threshold=-20, dt=dt)
# dlpotential(V_histories, dt=dt)
# plt.imshow(network.synaptic_weights.detach().numpy())
# plt.show()
