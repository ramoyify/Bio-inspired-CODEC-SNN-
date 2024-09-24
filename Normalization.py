import matplotlib.pyplot as plt
import numpy as np
import os

def create_normalised_3D_matrix(fig, times_red, senders_red, times_green, senders_green, times_blue, senders_blue, cols):
    rows = 320 * 240
    normalised_3Dmat = np.zeros((rows, cols, 3))
    for i in range(len(times_red)):
        normalised_3Dmat[int(senders_red[i]-1), int(np.round(times_red[i], 0)), 0] = 1.0  # red
    for i in range(len(times_green)):
        normalised_3Dmat[int(senders_green[i]-1), int(np.round(times_green[i], 0)), 1] = 1.0  # green
    for i in range(len(times_blue)):
        normalised_3Dmat[int(senders_blue[i]-1), int(np.round(times_blue[i], 0)), 2] = 1.0  # blue
    save_dir = os.path.join("npy_results", "normalized results")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"normalised_3Dmat_{fig}.npy")
    normalised_3Dmat.dump(file_path)
    return normalised_3Dmat.astype(bool)


def plot_results(fig, normalised_3Dmat, cols):
    """
    Plot the raster plot.
    Parameters:
        fig (str): Figure name.
        normalised_3Dmat (numpy.ndarray): Normalised 3D matrix of spike events.
        cols (int): Number of time bins.
    """
    # Print size
    print(f"normalised_3Dmat: {np.round((normalised_3Dmat.nbytes) / (1024 * 1024), 2)} MB")
    print()

    # Plot the raster plot
    plt.figure(figsize=(10, 6))
    for channel, color in enumerate(['red', 'green', 'blue']):
        print(f"Preparing plot for {color} ...")
        for idx in range(normalised_3Dmat.shape[0]):
            spike_times = np.where(normalised_3Dmat[idx, :, channel])[0] * (cols // 3)  # Adjust the indexing
            if len(spike_times) > 0:
                plt.vlines(spike_times, idx, idx + 1, color=color, linewidth=1)
    print("Done!")
    plt.xlabel('Time [ms]')
    plt.ylabel('Neuron Index')
    plt.title(f"Raster Plot of {fig}")
    plt.ylim(0, normalised_3Dmat.shape[0])
    plt.grid(True)
    save_dir = os.path.join("npy_results", "normalized results")
    plt.savefig(os.path.join(save_dir, f"{fig}_raster_plot.png"))
    # plt.show()


# List of figure names
figures = [f'frame{i:04d}' for i in range(1, 91)]  # Assuming your figures are named frame_0000, frame_0001, ...

# List colour names
colours = ["Red", "Green", "Blue"]
cols = 51  # time in ms

# Loop over each figure
for figure_name in figures:
    print(f"Processing {figure_name} ...")
    times = {}
    senders = {}

    for color in colours:
        # Load data for each color
        times[color] = np.load(f"npy_results/{figure_name}_{color}_ts.npy", allow_pickle=True)
        senders[color] = np.load(f"npy_results/{figure_name}_{color}_senders.npy", allow_pickle=True)

    # Create the normalized 2D matrix
    normalised_3Dmat = create_normalised_3D_matrix(figure_name, times["Red"], senders["Red"], times["Green"],
                                                   senders["Green"], times["Blue"], senders["Blue"], cols)

    # Print results and plot raster plot
    plot_results(figure_name, normalised_3Dmat, cols)
