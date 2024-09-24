import numpy as np
import nest
import matplotlib.pyplot as plt
from pydub import AudioSegment
import io
from scipy.io import wavfile
import os
from tqdm import tqdm  # Import tqdm for progress bar

# Reset the NEST kernel
nest.ResetKernel()
nest.set_verbosity(20)  # Set NEST verbosity level to 20

# Create a spike recorder
spike_recorder = nest.Create('spike_recorder')

# Initialize variables and lists
currents = []  # List to store input currents
spike_counts = []  # List to store spike counts
min_current = 0  # Variable to store the minimum current
inc = 1  # Increment value for increasing the current
current = 370  # Initial current value
num_spikes = 0  # Variable to store the number of spikes
neuron_params = {
    'C_m': 250.0,  # Membrane capacitance (pF)
    'tau_m': 10.0,  # Membrane time constant (ms)
    't_ref': 2.0,  # Refractory period (ms)
    'E_L': 0.0,  # Resting membrane potential (mV)
    'V_th': 20.0,  # Threshold potential (mV)
    'V_reset': 10.0,  # Reset potential (mV)
    'tau_syn_ex': 0.5,  # Excitatory synaptic time constant (ms)
    'tau_syn_in': 0.5  # Inhibitory synaptic time constant (ms)
}

# Create a single neuron with the IAF_PSC_ALPHA model
neuron = nest.Create('iaf_psc_alpha')

# List to store current and spike count pairs where the number of spikes increased
current_spikes_values = [[0, 0]]
current_spikes_idx = 0  # Index for current_spikes_values list
actual_number_spikes = 0  # Actual number of spikes observed
number_spikes = 0  # Number of spikes observed

# Connect the neuron to the spike recorder
nest.Connect(neuron, spike_recorder)

# Read the stereo sound file
audio_path = '/home/ntu-user/PycharmProjects/Assessment/Sound/extracted_audio.wav'
sound = AudioSegment.from_wav(audio_path)

# Split stereo sound into left and right channels
left_channel = sound.split_to_mono()[0]
right_channel = sound.split_to_mono()[1]

# Save left and right channels
left_channel_dir = 'Sound Data results/Left_Channel'
right_channel_dir = 'Sound Data results/Right_Channel'

os.makedirs(left_channel_dir, exist_ok=True)
os.makedirs(right_channel_dir, exist_ok=True)

# Function to chunk audio data with a progress bar
def chunk_audio_with_progress(data, rate, chunk_duration=5.0):
    chunk_size = int(chunk_duration * rate)
    num_chunks = len(data) // chunk_size

    # Use tqdm to create a progress bar
    for i in tqdm(range(num_chunks), desc='Chunking audio'):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        yield data[start:end]

# Chunk left and right channels with progress bar
chunk_duration = 5.0  # Adjust the chunk duration as needed
total_duration = len(left_channel) / 1000  # Total duration in seconds

num_chunks = 892  # Total number of chunks
chunk_duration = total_duration / num_chunks  # Adjusted chunk duration

# Using the modified chunk_audio_with_progress function
for i, (left_chunk, right_chunk) in enumerate(zip(chunk_audio_with_progress(left_channel.get_array_of_samples(), left_channel.frame_rate, chunk_duration),
                                                   chunk_audio_with_progress(right_channel.get_array_of_samples(), right_channel.frame_rate, chunk_duration))):
    # Save left and right channel chunks
    np.save(os.path.join(left_channel_dir, f'left_channel_chunk_{i}.npy'), np.array(left_chunk))
    np.save(os.path.join(right_channel_dir, f'right_channel_chunk_{i}.npy'), np.array(right_chunk))

    # Process each chunk here as needed
    # Your processing code goes here

    print(f"Processed chunk {i+1}/{num_chunks}")

# Now you can process each chunk individually
# Add your processing code here
