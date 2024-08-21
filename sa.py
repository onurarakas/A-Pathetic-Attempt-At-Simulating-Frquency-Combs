import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

plt.style.use('dark_background')


fr = 30  # Time between envelope pulses (spacing between envelope pulses)
num_pulses = 30  # Number of pulses
f = 2  # Frequency of the mode-locked pulses
t = np.linspace(-50, 700, 10000)  
phi = np.pi / 6

# Gaussian envelope for each pulse
def gaussian(t, center, width):
    #return np.exp(-((t - center)**2) / (2 * width**2))
    return np.exp(-((t - center)**2) / (width))


pulse_train = np.zeros_like(t)

# Generate the pulse train
for ii in range(num_pulses):
    # Calculate the center of each pulse idea 
    pulse_center = (t[0] + 5) + ii * fr
    # Add pulses to the pulse train
    pulse_train += gaussian(t, pulse_center, 5) * np.cos(2 * np.pi * f * t - ii * phi) #this how we made oscillations that are within an envelope
    #idea stolen from chatgpt

# Compute the envelope using the Hilbert transform, it is said that envelopes are calculated through hilber transforms no idea why
#analytic_signal = hilbert(pulse_train)
#envelope = np.abs(analytic_signal)

# fourier transformation stuff
pulse_train_freq_domain = np.fft.fft(pulse_train)
freq = np.fft.fftfreq(len(t), (t[1] - t[0]))


plt.figure(figsize=(20, 12))

# Time-domain plot
plt.subplot(2, 1, 1) #tek canvas a iki graph koyma
plt.plot(t, pulse_train, label='Mode-Locked Pulse Train', color = "orange")
#plt.plot(t, envelope, 'k--', label='Envelope')
#plt.plot(t, -envelope, 'k--')
plt.title('Mode-Locked Pulse Train and its Envelope')
plt.xlabel('Time (t)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()


plt.subplot(2, 1, 2)
plt.stem(freq, np.abs(pulse_train_freq_domain)/len(t), markerfmt=" ", basefmt="-b")
plt.xlim(-3.5,3.5)
plt.title('Frequency Domain Representation')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)


plt.tight_layout()
plt.show()

