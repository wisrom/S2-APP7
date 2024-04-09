import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import helpers as hp
import sounddevice as sd
def exemple1():
    duration = 10.0  # seconds
    fs = 44100  # sampling frequency
    f0 = 500  # initial frequency of the chirp
    f1 = 10000  # final frequency of the chirp
    t = np.linspace(0,duration,int(fs * duration))
    onde = signal.chirp(t,f0,duration,f1,method="linear")
    sd.play(onde,fs)
    sd.wait()
    # Plot the chirp signal
    #plt.figure()
    #plt.plot(t, onde)
    #plt.title('Linear Chirp Signal')
    #plt.xlabel('Time [s]')
    #plt.ylabel('Amplitude')
    #plt.grid(True)
    #plt.show()

def exemple2():
    duration = 10.0  # seconds
    fs = 11000  # sampling frequency
    f0 = 500  # initial frequency of the chirp
    f1 = 10000  # final frequency of the chirp
    t = np.linspace(0, duration, int(fs * duration))
    onde = signal.chirp(t, f0, duration, f1, method="linear")
    sd.play(onde, fs)
    sd.wait()

def exemple3():
    # Paramètres des signaux
    duration = 0.02  # 20 millisecondes
    frequency = 500  # 500 Hz

    # Temps pour le premier signal (fréquence d'échantillonnage de 50 kHz)
    fs1 = 50000  # Fréquence d'échantillonnage de 50 kHz
    t1 = np.linspace(0, duration, int(fs1 * duration))
    signal1 = np.sin(2 * np.pi * frequency * t1)

    # Temps pour le deuxième signal (fréquence d'échantillonnage de 600 Hz)
    fs2 = 600  # Fréquence d'échantillonnage de 600 Hz
    t2 = np.linspace(0, duration, int(fs2 * duration))
    signal2 = np.sin(2 * np.pi * frequency * t2)
    plt.figure()
    plt.plot(t1, signal1, label='Fréq. échant. = 50 kHz')
    plt.plot(t2, signal2, label='Fréq. échant. = 600 Hz')
    plt.title('Signaux Sinusoïdaux de 500 Hz avec différentes fréquences d\'échantillonnage')
    plt.xlabel('Temps (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()

def _exemple_2():
    duration = 10.0  # seconds
    fs = 44100  # sampling frequency
    f0 = 500  # initial frequency of the chirp
    f1 = 10000  # final frequency of the chirp
    t = np.linspace(0, duration, int(fs * duration))
    onde = signal.chirp(t, f0, duration, f1, method="linear")
    sd.play(onde, fs)
    sd.wait()
def _exemple_3():
    w = np.linspace(0,1,2000)
    s = 1j*w
    duration = 1
    Hs = 1/(np.sqrt(s**4+3.1239*s**3+4.3916*s**2+3.2011*s+1))
    b = [1.]
    a = [1, 4.4952, 9.6223, 12.3583, 9.9202, 4.6717, 1]
    t = np.linspace(0, duration, 2000)
    n = len(a) // 3
    r34 = np.roots(a)
    #a_parts = [[i:i + n] for i in range(0, len(a), n)]
    hp.bodeplot(b,a,"normalisé")
    r12 = np.roots(b)

    # Diviser les coefficients de a en trois parties égales
    n = len(a) // 3
    a_parts = [r34[i:i + n] for i in range(0, len(a), n)]




    p1 = np.poly(a_parts[0])
    p2 = np.poly(a_parts[1])
    p3 = np.poly(a_parts[2])
    #p5 = np.poly(a_parts[0],a_parts[1], a_parts[2])
    #p1 = np.poly(roots_set1)
   # p2 = np.poly(roots_set2)
    #p3 = np.poly(roots_set3)
    print(p1)
    print(p2)
    print(p3)
    Fc = 1500
    p1_denormalized = np.poly(np.roots(p1) * (2 * np.pi * Fc))
    p2_denormalized = np.poly(np.roots(p2) * (2 * np.pi * Fc))
    p3_denormalized = np.poly(np.roots(p3) * (2 * np.pi * Fc))
    result = np.convolve(np.convolve(p1_denormalized, p2_denormalized), p3_denormalized)
    b2 = [700852722000000000000000.]
    # Display the result
    print("Resulting polynomial:", result)
    #p4_denormalized = p1_denormalized * p2_denormalized * p3_denormalized
    #print(p4_denormalized)
    a_regrouped = p1_denormalized*p2_denormalized*p3_denormalized
    b1, a1 = p1_denormalized, p2_denormalized
    # Affichage des polynômes dénormalisés
    print("Polynôme p1 dénormalisé :", p1_denormalized)
    print("Polynôme p2 dénormalisé :", p2_denormalized)
    print("Polynôme p2 dénormalisé :", p3_denormalized)

    #b2, a2 = p1_denormalized, p2_denormalized
    hp.bodeplot(b2, result, "dénomralisé")

def bessel():
    order_range = range(1, 7)

    fs = 1
    for order in order_range:
        b, a = signal.bessel(order, 1, 'low', analog=True)
        w, h = signal.freqz(b, a, worN=8000)
        #plt.semilogx((fs * 0.5 / np.pi) * w, 20 * np.log10(abs(h)), label="Order {}".format(order))
        hp.bodeplot(b, a, "salut")



def main():
    #exemple1()
    order_range = range(1, 7)  # Range of filter orders from 1 to 6
    fs = 2800  # Sampling frequency
    fc = 1500  # Cutoff frequency (Hz)
    #exemple2()
    #exemple3()
    #_exemple_2()
    _exemple_3()
    #bessel()
    plt.show()
if __name__ == '__main__':
    main()