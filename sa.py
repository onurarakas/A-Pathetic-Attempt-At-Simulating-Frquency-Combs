import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from matplotlib.ticker import FormatStrFormatter

plt.style.use('dark_background')


Tr = 30  # Time between envelope pulses (spacing between envelope pulses)
num_pulses = 30  # Number of pulses
f = 2  # Frequency of the mode-locked pulses
t = np.linspace(-50, 3000, 100000)  
phi = np.pi / 6

# Gaussian envelope for each pulse
def gaussian(t, center, width):
    #return np.exp(-((t - center)**2) / (2 * width**2))
    return np.exp(-((t - center)**2) / (width))


pulse_train = np.zeros_like(t)

# Generate the pulse train
for ii in range(num_pulses):
    # Calculate the center of each pulse idea 
    pulse_center = (t[0] + 2) + ii * Tr
    # Add pulses to the pulse train
    pulse_train += gaussian(t, pulse_center, 5) * np.cos(2 * np.pi * f * t + ii * phi) #this how we made oscillations that are within an envelope
    #idea stolen from chatgpt

# Compute the envelope using the Hilbert transform, it is said that envelopes are calculated through hilber transforms no idea why
#analytic_signal = hilbert(pulse_train)
#envelope = np.abs(analytic_signal)

# fourier transformation stuff
pulse_train_freq_domain = np.fft.fft(pulse_train)
freq = np.fft.fftfreq(len(t), (t[1] - t[0]))

#burayı hızlıca chat gpt ye yaptırdım mantık basit fft aynı etkisi yapıyor ve negatif değerli frekanslarda printliyor onun önüne geçmek için fft verisinin sadece 2. yarısını kullanabiliriz
positive_freqs = freq[:len(freq) // 2]
positive_pulse_train_freq_domain = pulse_train_freq_domain[:len(pulse_train_freq_domain) // 2]


max_index = np.argmax(positive_pulse_train_freq_domain) # max_index gives the element of positive_pulse_train_freq_domain which has the highest value
print("which element of fft amplitude has the greatest value:\n", max_index,"greatest value of the amplitude(normalized)\n", np.abs(positive_pulse_train_freq_domain[max_index])/len(t))
#positive_pulse_train_freq_domain[max_index] gives the greatest value of the positive_pulse_train_freq_domain and when I do "np.abs(positive_pulse_train_freq_domain[max_index])/len(t))" I normalize the positive_pulse_train_freq_domain, with all these steps i get the greatest value of the amplitude which is seen in the graph

print("\nwhere greatest value of the amplitude lies in the x(freq) axis:\n", positive_freqs[max_index], "\n")


f_r = 1/Tr

f_0 = phi/(2*np.pi)*f_r

print("Repetition Rate Fr(1/Tr):\n",1/Tr,"\n" )

print("Phase-Envelope Offset f_0:\n", f_0, "\n")

print(f_r - f_0)


fig = plt.figure(figsize=(20, 12))


# Time-domain plot
plt.subplot(2, 1, 1) #tek canvas a iki graph koyma
plt.plot(t, pulse_train, label='Mode-Locked Pulse Train', color = "orange")
#plt.plot(t, envelope, 'k--', label='Envelope')
#plt.plot(t, -envelope, 'k--')
plt.title('Mode-Locked Pulse Train and its Envelope')
plt.xlabel('Time (t)')
plt.subplots_adjust(right=2.0)
plt.ylabel('Amplitude')
plt.grid(True)
plt.legend()

tick_positions2 = np.arange(positive_freqs[max_index], 4.0, f_r)
tick_positions1 = np.arange(positive_freqs[max_index], 0.0, -f_r)
tick_positions = np.concatenate((tick_positions1, tick_positions2))

print(tick_positions)

plt.subplot(2, 1, 2)
#plt.stem(freq, np.abs(pulse_train_freq_domain)/len(t), markerfmt=" ", basefmt="-b")
plt.stem(positive_freqs, np.abs(positive_pulse_train_freq_domain)/len(t), markerfmt=" ")
plt.xticks(tick_positions)
# Format ticks to show 5 decimal places
ax = plt.gca()
ax.xaxis.set_major_formatter(FormatStrFormatter('%.9f'))

#plt.suptitle('Function Formatting', fontsize=16, x=1/Tr, ha='right')
#fig1.suptitle('Function Formatting', fontsize=16, x=, ha='left')
plt.xlim(0,4)
#plt.ylim(0, np.abs(positive_pulse_train_freq_domain[max_index])/len(t))

plt.title('Frequency Domain Representation')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(False)


plt.tight_layout()
plt.show()
