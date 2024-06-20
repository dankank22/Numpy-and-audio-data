
# Lab 2 Report

import numpy as np

from IPython.display import Image #For displaying images in colab jupyter cell

# Exercise 1: Loops vs Numpy operations

Image('exercise1.PNG', width = 1000)

arr2d_1 = np.random.randn(1000, 1000) * 10
arr2d_2 = np.random.randn(1000, 1000) * 10

import time # Import time to measure computational efficiency of the code

# Elementwise addition using loop

arr2d_3_loop = np.zeros((1000, 1000)) # Create a placeholder array for arr2d_3
[length, height] = arr2d_3_loop.shape

start_time_loop = time.time() # start time of the code

for i in range (0, length):
    for j in range (0, height):
        arr2d_3_loop[i, j] = arr2d_1[i, j] + arr2d_2[i, j]

        end_time_loop = time.time() # end time of the code

elapsed_time_loop = end_time_loop - start_time_loop # end time - start time -> elapsed time in seconds
print(elapsed_time_loop)

# Elementwise addition using Numpy function

start_time_np = time.time()

arr2d_3_np = np.add(arr2d_1, arr2d_2)

end_time_np = time.time()

elapsed_time_np = end_time_np - start_time_np
print(elapsed_time_np)

# Make sure two outputs are equivalent

np.sum(arr2d_3_loop == arr2d_3_np) == 1000 * 1000 # Should output True if the outputs are same

### Which computation is faster and by what factor?
### e.g. a code that takes 0.1s is faster by a factor of 10 compared to a code that takes 1s

The elementwise addition using the Numpy function is faster by a factor of greater than 250.

# Exercise 2: Generate Triangular Waveform

Image('exercise2.PNG', width = 1000)

import matplotlib.pyplot as plt

signal_range = [0, 6]
sampling_frequency = 10
# Creates x values
x_values = np.arange(signal_range[0], signal_range[1], 1 / sampling_frequency)
# Creates an array of zeros the same size as x values

y_values = np.zeros_like(x_values)
# Initializing first x vale
x = signal_range[0]
index = 0
# determines if the first signal bound is even, odd, or zero
if signal_range[0] == 0:
    y = -1.0
elif signal_range[0] % 2 == 0:
    y = -1.0
elif signal_range[0] % 1 == 0:
    y = 1.0
# sets the slope depending on if y is a positive or negative integer value
for i in range (signal_range[0] * sampling_frequency, signal_range[1] * sampling_frequency):
    # print(round(x, 5))
    if y == -1.0:
        dy = 2 / sampling_frequency
    elif y == 1.0:
        dy = - 2 / sampling_frequency
    y_values[index] = round(y, 5)
    y += dy
    dx = 1 / sampling_frequency
    index += 1
    x += dx
    # print(y_values[i])
plt.stem(x_values, y_values)
plt.plot(x_values, y_values, c="blue")


# Exercise 3: Sinusoidal Generator

Image('exercise3.PNG', width = 1000)

# Define generate_sine function
import math

def generate_sine(t_duration, f0, fs):
    t_arr = np.arange(0, t_duration, 1 / fs)
    amplitudes = np.zeros(t_arr.size) # 
    for i in range(0, t_duration * fs):
        amplitudes[i] = math.sin(2 * math.pi * f0 * t_arr[i])
    
    # Return 1D numpy arrays each containing timepoints and sine waveform amplitudes
    return t_arr, amplitudes 
plot_values = generate_sine(10, 0.5, 10)
# plt.plot(plot_values[0], plot_values[1])

# parameter set 1
t_duration_1 = 5
f0_1 = 0.5
fs_1 = 100
t_arr_1, amplitudes_1 = generate_sine(t_duration_1, f0_1, fs_1)

# parameter set 2
t_duration_2 = 5
f0_2 = 1.
fs_2 = 100
t_arr_2, amplitudes_2 = generate_sine(t_duration_2, f0_2, fs_2)

# parameter set 3
t_duration_3 = 5
f0_3 = 1.5
fs_3 = 100
t_arr_3, amplitudes_3 = generate_sine(t_duration_3, f0_3, fs_3)

# Plot 3 x 1 subplot showing all three waveform
import matplotlib.pyplot as plt
# Create supplot for Wave Frequency 0.5
plt.subplot(311)
plt.plot(t_arr_1, amplitudes_1)
plt.title('Wave Frequency of 0.5')
plt.xlabel('t')
plt.ylabel('amplitude')
# Create supplot for Wave Frequency 1.0
fig = plt.figure(2)
plt.subplot(312)
plt.plot(t_arr_2, amplitudes_2)
plt.title('Wave Frequency of 1.0')
plt.xlabel('t')
plt.ylabel('amplitude')
# Create supplot for Wave Frequency 1.5
plt.subplot(313)
plt.plot(t_arr_3, amplitudes_3)
plt.title('Wave Frequency of 1.5')
plt.xlabel('t')
plt.ylabel('amplitude')
# Format graphs correctly
fig.tight_layout() 

# Exercise 4: Notes Synthesis

Image('exercise4.PNG', width = 1000)

import IPython.display as ipd
from scipy.io import wavfile as wav

sampling_rate = 8000
t_duration = 1
amplitude = 1
initial_frequency = 220
# Calling generate_sine function for 8 different frequencies (in Hz)
note_1 = generate_sine(t_duration, initial_frequency * 2 ** (0/12), sampling_rate)
note_2 = generate_sine(t_duration, initial_frequency * 2 ** (2/12), sampling_rate)
note_3 = generate_sine(t_duration, initial_frequency * 2 ** (4/12), sampling_rate)
note_4 = generate_sine(t_duration, initial_frequency * 2 ** (5/12), sampling_rate)
note_5 = generate_sine(t_duration, initial_frequency * 2 ** (7/12), sampling_rate)
note_6 = generate_sine(t_duration, initial_frequency * 2 ** (9/12), sampling_rate)
note_7 = generate_sine(t_duration, initial_frequency * 2 ** (11/12), sampling_rate)
note_8 = generate_sine(t_duration, initial_frequency * 2 ** (12/12), sampling_rate)
# Concatenating 8 notes together and multiplying by 32767
a_major_scale = 32767 * np.concatenate([note_1[1], note_2[1], note_3[1], note_4[1], note_5[1], note_6[1], note_7[1], note_8[1]])
audio_file_name = 'a_major_scale.wav'
# Writes the wav file using file name, sampling rate, and amplitude data 
a_major_scale = wav.write(audio_file_name, sampling_rate, a_major_scale.astype('int16'))
display(ipd.Audio(audio_file_name))

# NOTE: Multiply your concatenated notes (with amplitude of 1) with 32767 followed by conversion to int16 format
# before playing or writing your audio array into a file. 

# Exercise 5: Chord Synthesis

Image('exercise5.PNG', width = 1000)

sampling_rate = 8000
t_duration = 1
amplitude = 1
initial_frequency = 220
note_1 = generate_sine(t_duration, initial_frequency * 2 ** (0/12), sampling_rate)
note_2 = generate_sine(t_duration, initial_frequency * 2 ** (2/12), sampling_rate)
note_3 = generate_sine(t_duration, initial_frequency * 2 ** (4/12), sampling_rate)
note_4 = generate_sine(t_duration, initial_frequency * 2 ** (5/12), sampling_rate)
note_5 = generate_sine(t_duration, initial_frequency * 2 ** (7/12), sampling_rate)
note_6 = generate_sine(t_duration, initial_frequency * 2 ** (9/12), sampling_rate)
note_7 = generate_sine(t_duration, initial_frequency * 2 ** (11/12), sampling_rate)
note_8 = generate_sine(t_duration, initial_frequency * 2 ** (12/12), sampling_rate)
# Adding the two amplitudes
chord_1 = np.add(note_1[1], note_3[1])
chord_2 = np.add(note_2[1], note_4[1])
chord_3 = np.add(note_3[1], note_5[1])
chord_4 = np.add(note_4[1], note_6[1])
chord_5 = np.add(note_5[1], note_7[1])
chord_6 = np.add(note_6[1], note_8[1])
# Concatenating normalized notes
chords = 32767 * np.concatenate([chord_1 / np.max(chord_1), chord_2 / np.max(chord_2), chord_3 / np.max(chord_3), chord_4 / np.max(chord_4), chord_5 / np.max(chord_5), chord_6 / np.max(chord_6)])
audio_file_name = 'chords.wav'
# Writes the wav file using file name, sampling rate, and amplitude data 
chords = wav.write(audio_file_name, sampling_rate, chords.astype('int16'))
display(ipd.Audio(audio_file_name))

# NOTE: Multiply your concatenated notes (with amplitude of 1) with 32767 followed by conversion to int16 format
# before playing or writing your audio array into a file. 
