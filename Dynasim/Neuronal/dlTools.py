import matplotlib.pyplot as plt
import numpy as np


def dlraster(V_histories, threshold=-20, dt=0.01):

    V_histories = V_histories.cpu().detach().numpy()  # Convert to numpy array if it's a torch tensor
    timesteps, num_neurons = V_histories.shape

    spike_times = []
    neuron_ids = []

    for neuron_id in range(num_neurons):
        spikes = np.where(V_histories[:, neuron_id] > threshold)[0]
        spike_times.extend(spikes*dt)
        neuron_ids.extend([neuron_id] * len(spikes))

    plt.figure(figsize=(10, 6))
    plt.scatter(spike_times, neuron_ids, s=1, color='black', marker='o')

    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.title('Raster Plot')
    plt.show()


def dlpotential(V_histories, dt=0.01):

    image = V_histories.cpu().detach().numpy().transpose()  # Convert to numpy array if it's a torch tensor
    timesteps, num_neurons = V_histories.shape
    xaxis = np.linspace(0, dt*timesteps, timesteps)
    yaxis = np.linspace(1, num_neurons, num_neurons)

    plt.figure(figsize=(10, 10))
    extent = [xaxis[0], xaxis[-1], yaxis[0], yaxis[-1]]

    # Display the image
    plt.imshow(image, extent=extent, aspect='auto', origin='upper')
    plt.colorbar()
    plt.clim(-90, 0)

    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron ID')
    plt.title('Membrane potential (mV)')
    plt.show()


def dlConnections(net):

    plt.imshow(net.synaptic_weights.cpu().detach().numpy())
    plt.colorbar()

    plt.xlabel("Sync")
    plt.ylabel("Source")
    plt.title("Synaptic Connections' strength")
    plt.show()

    return

