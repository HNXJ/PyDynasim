import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim


class Dynalearn:

    def __init__(self, network, lr=1.0):

        self.network = network
        self.optimizer = optim.SGD(self.network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def optimize(self, I_ext, target_firing_rate, num_epochs=100, time_sim=100):

        for epoch in range(num_epochs):

            V_histories = self.network(I_ext, T=time_sim)
            firing_rates = self.compute_firing_rate(V_histories)
            loss = self.criterion(firing_rates, target_firing_rate)
            print(firing_rates, target_firing_rate)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.network.train()

            # if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')

    def compute_firing_rate(self, V_histories, threshold=-20):

        spikes = (V_histories > threshold).float().sum(dim=0)
        firing_rates = nn.Parameter(spikes.sum(dim=0) / V_histories.shape[0])
        return firing_rates

    def compute_mean_membrane_potential(self, V_histories, threshold=0):

        spikes = torch.mean(V_histories, dim=0)
        mean_potential = nn.Parameter(spikes.mean(dim=0))
        return mean_potential
