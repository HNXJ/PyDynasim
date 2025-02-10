import torch
import torch.nn as nn


class HodgkinHuxleyNeuron(nn.Module):
    def __init__(self, num_neurons, dt=0.1, inoise=10, device=torch.device(1)):
        super(HodgkinHuxleyNeuron, self).__init__()
        self.num_neurons = num_neurons
        self.dt = dt
        self.device = device
        self.I_noise = inoise

        # Hodgkin-Huxley constants
        self.C_m = torch.Tensor([1.0]).requires_grad_(False).to(self.device)
        self.g_Na = torch.Tensor([120.0]).requires_grad_(False).to(self.device)
        self.g_K = torch.Tensor([36.0]).requires_grad_(False).to(self.device)
        self.g_L = torch.Tensor([0.3]).requires_grad_(False).to(self.device)
        self.E_Na = torch.Tensor([50.0]).requires_grad_(False).to(self.device)
        self.E_K = torch.Tensor([-77.0]).requires_grad_(False).to(self.device)
        self.E_L = torch.Tensor([-54.387]).requires_grad_(False).to(self.device)

        # Initial states
        self.V = torch.Tensor(torch.Tensor(torch.zeros(num_neurons) - torch.rand(num_neurons)*70)).requires_grad_(False).to(self.device)
        self.m = torch.Tensor(torch.zeros(num_neurons)).requires_grad_(False).to(self.device)
        self.h = torch.Tensor(torch.zeros(num_neurons)).requires_grad_(False).to(self.device)
        self.n = torch.Tensor(torch.zeros(num_neurons)).requires_grad_(False).to(self.device)

    def alpha_m(self, V): return torch.Tensor((2.5 - 0.1 * (V + 65)) / (torch.exp(2.5 - 0.1 * (V + 65)) - 1))
    def beta_m(self, V): return torch.Tensor(4.0 * torch.exp(-(V + 65) / 18))
    def alpha_h(self, V): return torch.Tensor(0.07 * torch.exp(-(V + 65) / 20))
    def beta_h(self, V): return torch.Tensor(1.0 / (torch.exp(3.0 - 0.1 * (V + 65)) + 1))
    def alpha_n(self, V): return torch.Tensor((0.1 - 0.01 * (V + 65)) / (torch.exp(1.0 - 0.1 * (V + 65)) - 1))
    def beta_n(self, V): return torch.Tensor(0.125 * torch.exp(-(V + 65) / 80))

    def forward(self, I_ext, I_synaptic):

        # Update gating variables
        self.m = torch.add(self.m, self.alpha_m(self.V) * (1 - self.m) - self.beta_m(self.V) * self.m, alpha=self.dt).nan_to_num(nan=0).clamp(min=-10, max=10)
        self.h = torch.add(self.h, self.alpha_h(self.V) * (1 - self.h) - self.beta_h(self.V) * self.h, alpha=self.dt).nan_to_num(nan=0).clamp(min=-10, max=10)
        self.n = torch.add(self.n, self.alpha_n(self.V) * (1 - self.n) - self.beta_n(self.V) * self.n, alpha=self.dt).nan_to_num(nan=0).clamp(min=-10, max=10)

        # Compute currents
        I_Na = self.g_Na * self.m**3 * self.h * (self.V - self.E_Na)
        I_K = self.g_K * self.n**4 * (self.V - self.E_K)
        I_L = self.g_L * (self.V - self.E_L)

        I_noise = (torch.randn(I_synaptic.size()) * self.I_noise).to(self.device)
        I_total = torch.add(I_ext + I_noise, I_synaptic + I_Na + I_K + I_L, alpha=-1.0)

        # Update membrane potential
        self.V = torch.div(torch.add(self.V, I_total, alpha=self.dt), self.C_m).nan_to_num(nan=-70, posinf=-60, neginf=-70).clamp(min=-77, max=40)

        return self.V


class HHNeurons(nn.Module):
    def __init__(self, num_neurons, population_type, connections, dt=0.1, inoise=10, device=torch.device(1)):
        super(HHNeurons, self).__init__()
        self.num_neurons = num_neurons
        self.population_type = population_type
        self.connections = connections
        self.dt = dt

        self.device = device
        self.num_neurons_total = sum(num_neurons)
        self.neurons = HodgkinHuxleyNeuron(self.num_neurons_total, dt=dt, inoise=inoise, device=self.device).to(self.device)
        self.synapse_types = torch.zeros(self.num_neurons_total, self.num_neurons_total, dtype=torch.int32).requires_grad_(False)

        self.synapse_type_map = {
            'AMPA': 1,
            'NMDA': 2,
            'GABAa': 3,
            'GABAb': 4,
            'ACh': 5,
            '5HT': 6,
            'DA': 7,
            'NE': 8,
            'Q': 9,
            'D1' : 10
        }

        # Synaptic conductances and reversal potentials
        self.g_AMPA = torch.Tensor([0.25]).requires_grad_(True).to(self.device)
        self.g_GABAa = torch.Tensor([0.25]).requires_grad_(False).to(self.device)
        self.g_GABAb = torch.Tensor([0.25]).requires_grad_(False).to(self.device)
        self.g_NMDA = torch.Tensor([0.25]).requires_grad_(False).to(self.device)
        self.g_ACh = torch.Tensor([0.25]).requires_grad_(False).to(self.device)
        self.g_D1 = torch.Tensor([0.25]).requires_grad_(False).to(self.device)

        self.E_AMPA = torch.Tensor([0.0]).requires_grad_(False).to(self.device)
        self.E_GABAa = torch.Tensor([-70.0]).requires_grad_(False).to(self.device)
        self.E_GABAb = torch.Tensor([-90.0]).requires_grad_(False).to(self.device)
        self.E_NMDA = torch.Tensor([0.0]).requires_grad_(False).to(self.device)
        self.E_ACh = torch.Tensor([0.0]).requires_grad_(False).to(self.device)
        self.E_D1 = torch.Tensor([0.0]).requires_grad_(False).to(self.device)

        self.s_AMPA = torch.zeros(self.num_neurons_total).requires_grad_(False).to(self.device)
        self.s_GABAa = torch.zeros(self.num_neurons_total).requires_grad_(False).to(self.device)
        self.s_GABAb = torch.zeros(self.num_neurons_total).requires_grad_(False).to(self.device)
        self.s_NMDA = torch.zeros(self.num_neurons_total).requires_grad_(False).to(self.device)
        self.s_ACh = torch.zeros(self.num_neurons_total).requires_grad_(False).to(self.device)
        self.s_D1 = torch.zeros(self.num_neurons_total).requires_grad_(False).to(self.device)

        # Create connection matrices
        self.synaptic_weights = nn.Parameter(self.merge_connections(), requires_grad=True)

    def merge_connections(self):

        w = torch.zeros(self.num_neurons_total, self.num_neurons_total, dtype=torch.float32)

        for (src_id, dst_id, synapse_type, weight_matrix) in self.connections:
            src_start = sum(self.num_neurons[:src_id])
            src_end = src_start + self.num_neurons[src_id]
            dst_start = sum(self.num_neurons[:dst_id])
            dst_end = dst_start + self.num_neurons[dst_id]

            w[dst_start:dst_end, src_start:src_end] = weight_matrix
            self.synapse_types[dst_start:dst_end, src_start:src_end] = self.synapse_type_map[synapse_type]

        return w

    def forward(self, I_ext, T):

        V_histories = []

        # Initial membrane potentials
        V = self.neurons.V.clone().to(self.device)

        # Simulation loop
        num_steps = int(T / self.dt)

        for _ in range(num_steps):

            I_syn = torch.matmul(self.synaptic_weights, V)

            AMPA_mask = (self.synapse_types == self.synapse_type_map['AMPA']).float().to(self.device)
            GABAa_mask = (self.synapse_types == self.synapse_type_map['GABAa']).float().to(self.device)
            GABAb_mask = (self.synapse_types == self.synapse_type_map['GABAb']).float().to(self.device)
            NMDA_mask = (self.synapse_types == self.synapse_type_map['NMDA']).float().to(self.device)
            ACh_mask = (self.synapse_types == self.synapse_type_map['ACh']).float().to(self.device)
            D1_mask = (self.synapse_types == self.synapse_type_map['D1']).float().to(self.device)

            AMPA_input = torch.sum(I_syn * AMPA_mask, dim=1)
            GABAa_input = torch.sum(I_syn * GABAa_mask, dim=1)
            GABAb_input = torch.sum(I_syn * GABAb_mask, dim=1)
            NMDA_input = torch.sum(I_syn * NMDA_mask, dim=1)
            ACh_input = torch.sum(I_syn * ACh_mask, dim=1)
            D1_input = torch.sum(I_syn * D1_mask, dim=1)

            self.s_AMPA = torch.add(self.s_GABAa, -self.s_AMPA / 2.0, alpha=self.dt)  # TauD = 2.0, decay time constant
            self.s_GABAa = torch.add(self.s_GABAa, - self.s_GABAa / 5.0, alpha=self.dt)
            self.s_GABAb = torch.add(self.s_GABAb, - self.s_GABAb / 10.0, alpha=self.dt)
            self.s_NMDA = torch.add(self.s_NMDA, - self.s_NMDA / 100.0, alpha=self.dt)
            self.s_ACh = torch.add(self.s_ACh, - self.s_ACh / 50.0, alpha=self.dt)
            self.s_D1 = torch.add(self.s_D1, - self.s_D1 / 30.0, alpha=self.dt)

            self.s_AMPA = torch.matmul(self.synaptic_weights, self.s_AMPA)
            self.s_GABAa = torch.matmul(self.synaptic_weights, self.s_GABAa)
            self.s_GABAb = torch.matmul(self.synaptic_weights, self.s_GABAb)
            self.s_NMDA = torch.matmul(self.synaptic_weights, self.s_NMDA)
            self.s_ACh = torch.matmul(self.synaptic_weights, self.s_ACh)
            self.s_D1 = torch.matmul(self.synaptic_weights, self.s_D1)

            I_AMPA = torch.mul(self.g_AMPA * self.s_AMPA * (V - self.E_AMPA), AMPA_input)
            I_GABAa = torch.mul(self.g_GABAa * self.s_GABAa * (V - self.E_GABAa), GABAa_input)
            I_GABAb = torch.mul(self.g_GABAb * self.s_GABAb * (V - self.E_GABAb), GABAb_input)
            I_NMDA = torch.mul(self.g_NMDA * self.s_NMDA * (V - self.E_NMDA), NMDA_input)
            I_ACh = torch.mul(self.g_ACh * self.s_ACh * (V - self.E_ACh), ACh_input)
            I_D1 = torch.mul(self.g_D1 * self.s_D1 * (V - self.E_D1), D1_input)

            I_synaptic = I_AMPA + I_GABAa + I_GABAb + I_NMDA + I_ACh + I_D1

            V = self.neurons(I_ext, I_synaptic)
            V_histories.append(V.unsqueeze(0))

        return torch.cat(V_histories, dim=0)
