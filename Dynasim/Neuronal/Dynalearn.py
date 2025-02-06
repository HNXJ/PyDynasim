import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.optim import Optimizer
import torch.distributions as dist

class Dynalearn:

    def __init__(self, network, lr=1.0):

        self.network = network
        self.optimizer = GeneticSGD(self.network.parameters(), lr=lr)
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


class GeneticSGD(Optimizer):
    def __init__(self, params, lr=0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, gsdr=True, rdelta=0.1, ralpha=0.9):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov,
                        gsdr=gsdr, rdelta=rdelta, ralpha=ralpha)
        super(GeneticSGD, self).__init__(params, defaults)
        self.best_loss = float('inf')
        self.best_params = None

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']
            gsdr = group['gsdr']
            rdelta = group['rdelta']
            ralpha = group['ralpha']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                if gsdr:
                    normal = dist.Normal(0, 1)
                    random_normal = normal.sample(d_p.size())
                    d_p = d_p.add(ralpha * rdelta * random_normal, d_p * (1 - ralpha))

                p.data.add_(-group['lr'], d_p)

        if loss is None:

            return loss

        elif loss < self.best_loss:

            self.best_loss = loss
            self.best_params = [p.clone().detach() for group in self.param_groups for p in group['params']]

        return loss
