import torch
import torch.nn as nn


class HodgkinHuxleyNeuron(nn.Module):
    def __init__(self, num_neurons, dt=0.01):
        super(HodgkinHuxleyNeuron, self).__init__()
        self.num_neurons = num_neurons
        self.dt = dt

        # Hodgkin-Huxley constants
        self.C_m = 1.0
        self.g_Na = 120.0
        self.g_K = 36.0
        self.g_L = 0.3
        self.E_Na = 50.0
        self.E_K = -77.0
        self.E_L = -54.387

        # Initial states
        self.V = torch.zeros(num_neurons)
        self.m = torch.zeros(num_neurons)
        self.h = torch.zeros(num_neurons)
        self.n = torch.zeros(num_neurons)

    def alpha_m(self, V): return (2.5 - 0.1 * V) / (torch.exp(2.5 - 0.1 * V) - 1)
    def beta_m(self, V): return 4.0 * torch.exp(-V / 18)
    def alpha_h(self, V): return 0.07 * torch.exp(-V / 20)
    def beta_h(self, V): return 1.0 / (torch.exp(3.0 - 0.1 * V) + 1)
    def alpha_n(self, V): return (0.1 - 0.01 * V) / (torch.exp(1.0 - 0.1 * V) - 1)
    def beta_n(self, V): return 0.125 * torch.exp(-V / 80)

    def forward(self, I_pre):

        # Update gating variables
        self.m = self.m + self.dt * (self.alpha_m(self.V) * (1 - self.m) - self.beta_m(self.V) * self.m)
        self.h = self.h + self.dt * (self.alpha_h(self.V) * (1 - self.h) - self.beta_h(self.V) * self.h)
        self.n = self.n + self.dt * (self.alpha_n(self.V) * (1 - self.n) - self.beta_n(self.V) * self.n)

        # Compute currents
        I_Na = self.g_Na * self.m**3 * self.h * (self.V - self.E_Na)
        I_K = self.g_K * self.n**4 * (self.V - self.E_K)
        I_L = self.g_L * (self.V - self.E_L)

        # Update membrane potential
        self.V = self.V + self.dt * (I_pre - I_Na - I_K - I_L) / self.C_m
        self.V = torch.clamp(self.V, -100, 100)

        return self.V.nan_to_num()


class HHNeurons(nn.Module):
    def __init__(self, num_neurons, population_type, connections, dt=0.01):
        super(HHNeurons, self).__init__()
        self.num_neurons = num_neurons
        self.population_type = population_type
        self.connections = connections
        self.dt = dt

        self.num_neurons_total = sum(num_neurons)
        self.neurons = HodgkinHuxleyNeuron(self.num_neurons_total, dt=dt)

        self.synaptic_weights = torch.zeros(self.num_neurons_total, self.num_neurons_total)
        self.synapse_types = torch.zeros(self.num_neurons_total, self.num_neurons_total, dtype=torch.int32)

        self.synapse_type_map = {
            'AMPA': 1,
            'GABAa': 2,
            'GABAb': 3,
            'NMDA': 4,
            'ACh': 5,
            'D1': 6
        }

        # Synaptic conductances and reversal potentials
        self.g_AMPA = 0.1
        self.g_GABAa = 0.1
        self.g_GABAb = 0.1
        self.g_NMDA = 0.1
        self.g_ACh = 0.1
        self.g_D1 = 0.1

        self.E_AMPA = 0.0
        self.E_GABAa = -70.0
        self.E_GABAb = -90.0
        self.E_NMDA = 0.0
        self.E_ACh = 0.0
        self.E_D1 = 0.0

        self.s_AMPA = torch.zeros(self.num_neurons_total)
        self.s_GABAa = torch.zeros(self.num_neurons_total)
        self.s_GABAb = torch.zeros(self.num_neurons_total)
        self.s_NMDA = torch.zeros(self.num_neurons_total)
        self.s_ACh = torch.zeros(self.num_neurons_total)
        self.s_D1 = torch.zeros(self.num_neurons_total)

        # Create connection matrices
        self.create_connections()

    def create_connections(self):
        for (src_id, dst_id, synapse_type, weight_matrix) in self.connections:
            src_start = sum(self.num_neurons[:src_id])
            src_end = src_start + self.num_neurons[src_id]
            dst_start = sum(self.num_neurons[:dst_id])
            dst_end = dst_start + self.num_neurons[dst_id]

            self.synaptic_weights[dst_start:dst_end, src_start:src_end] = weight_matrix
            self.synapse_types[dst_start:dst_end, src_start:src_end] = self.synapse_type_map[synapse_type]

    def forward(self, I_ext, T):
        V_histories = []

        # Initial membrane potentials
        V = self.neurons.V.clone()

        # Simulation loop
        num_steps = int(T / self.dt)
        for _ in range(num_steps):
            I_syn = torch.matmul(self.synaptic_weights, V)

            AMPA_mask = (self.synapse_types == self.synapse_type_map['AMPA']).float()
            GABAa_mask = (self.synapse_types == self.synapse_type_map['GABAa']).float()
            GABAb_mask = (self.synapse_types == self.synapse_type_map['GABAb']).float()
            NMDA_mask = (self.synapse_types == self.synapse_type_map['NMDA']).float()
            ACh_mask = (self.synapse_types == self.synapse_type_map['ACh']).float()
            D1_mask = (self.synapse_types == self.synapse_type_map['D1']).float()

            # Update synaptic gating variables
            self.s_AMPA = self.s_AMPA + self.dt * (-self.s_AMPA / 2.0)  # Example decay term (TauD)
            self.s_GABAa = self.s_GABAa + self.dt * (-self.s_GABAa / 5.0)
            self.s_GABAb = self.s_GABAb + self.dt * (-self.s_GABAb / 10.0)
            self.s_NMDA = self.s_NMDA + self.dt * (-self.s_NMDA / 100.0)
            self.s_ACh = self.s_ACh + self.dt * (-self.s_ACh / 50.0)
            self.s_D1 = self.s_D1 + self.dt * (-self.s_D1 / 30.0)

            I_AMPA = torch.mul(self.g_AMPA * self.s_AMPA * (V - self.E_AMPA), AMPA_mask * self.synaptic_weights).sum()
            I_GABAa = torch.mul(self.g_GABAa * self.s_GABAa * (V - self.E_GABAa), GABAa_mask * self.synaptic_weights).sum()
            I_GABAb = torch.mul(self.g_GABAb * self.s_GABAb * (V - self.E_GABAb), GABAb_mask * self.synaptic_weights).sum()

            I_NMDA = torch.mul(self.g_NMDA * self.s_NMDA * (V - self.E_NMDA), NMDA_mask * self.synaptic_weights).sum()
            I_ACh = torch.mul(self.g_ACh * self.s_ACh * (V - self.E_ACh), ACh_mask * self.synaptic_weights).sum()
            I_D1 = torch.mul(self.g_D1 * self.s_D1 * (V - self.E_D1), D1_mask * self.synaptic_weights).sum()

            V = self.neurons(I_ext - I_AMPA + I_GABAa + I_GABAb - I_NMDA - I_ACh - I_D1)
            V_histories.append(V.unsqueeze(0))

        return torch.cat(V_histories, dim=0)
