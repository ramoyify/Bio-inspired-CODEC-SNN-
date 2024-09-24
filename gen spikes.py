import numpy as np
import librosa
import nest

# Load the audio file
audio_file = 'Sound/extracted_audio.wav'
audio_data, sample_rate = librosa.load(audio_file, sr=None, mono=True)

# Normalize the audio data to fit within a desired current range
min_current = 0.0
max_current = 100.0
normalized_audio = np.interp(audio_data, (audio_data.min(), audio_data.max()), (min_current, max_current))

# Reset the NEST kernel
nest.ResetKernel()
nest.set_verbosity(20)  # Set NEST verbosity level to 20

# Create a spike recorder
spike_recorder = nest.Create('spike_recorder')

# Create a single neuron with the IAF_PSC_ALPHA model
neuron = nest.Create('iaf_psc_alpha')

# Connect the neuron to the spike recorder
nest.Connect(neuron, spike_recorder)

# Set the input current to the neuron based on the normalized audio data
current_values = normalized_audio.tolist()
nest.SetStatus(neuron, {'I_e': current_values})

# Simulate for the duration of the audio
simulation_time = len(audio_data) / sample_rate * 1000  # Convert audio duration to milliseconds
nest.Simulate(simulation_time)

# Get the spike times
spike_times = nest.GetStatus(spike_recorder, 'events')[0]['times']

# Save the spike times to a file for further analysis
np.save('SoundSpikes/audio_spike_times.npy', spike_times)
