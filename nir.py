import numpy as np
import matplotlib.pyplot as pl
from scipy import optimize, special, interpolate, signal
from scipy.signal import hilbert, butter, lfilter, sosfilt

import params


def main():
    #Load experimental pulse-shape
    pulseTime = np.loadtxt("NIR-Impuls/NIR_time.dat")

    #Generate plots for time- and fourier-space
    fig, (axT, axF) = pl.subplots(1, 2)
    
    #Recalculate into atomic units
    fs_conv         = params.fs_conv
    E_conv          = params.E_conv
    THz_conv        = params.THz_conv
    pulseTime[:,0]  *= fs_conv*1e15
    pulseTime[:,1]  *= E_conv*1e-6
    time            = pulseTime[:,0]
    axT.plot(pulseTime[:,0], pulseTime[:,1], "g.", label="original pulse")

    #Calculation of the fouriertransform
    dt          = pulseTime[0,0] - pulseTime[1,0]
    freq        = np.fft.fftshift(np.fft.fftfreq(np.size(time), d=dt) )
    fourier     = np.abs(np.fft.fftshift(np.fft.fft(pulseTime[:,1], norm="ortho") ) )
    peaks       = signal.find_peaks(fourier)[0]
    axF.plot(freq, fourier, "b-", label = "fourier-transform of pulse")
    axF.plot(freq[peaks[1]], fourier[peaks[1]], "gx", label = "fourier-transform of pulse")

    print(freq[peaks] )
    #Filter the fast frequency out of the data
    k           = 1
    sos         = butter(10, 2*freq[peaks[k] ], 'lp', fs=10*freq[peaks[k] ], output='sos')
    filSig      = sosfilt(sos, pulseTime[:,1])
    axT.plot(time, filSig, "b.", label="filtered pulse")

    #Calculation of the envelope
    envelope    = np.abs(hilbert(filSig ) )
    #b, a        = butter(2, freq[peaks[1] ]*2, btype="low", analog=False)
    #envelope    = lfilter(b, a, envelope)
    peaks       = signal.find_peaks(envelope)[0]
    envelopeFct = interpolate.interp1d(time, envelope, kind="cubic")
    axT.plot(time, envelope, "b-", label="envelope")
    axT.plot(pulseTime[peaks,0], envelope[peaks], "x")

    #Fourier of envelope
    dt          = pulseTime[0,0] - pulseTime[1,0]
    freq        = np.fft.fftshift(np.fft.fftfreq(np.size(time), d=dt) )
    fourier     = np.abs(np.fft.fftshift(np.fft.fft(envelope, norm="ortho") ) )
    axF.plot(freq, fourier, "g-", label = "fourier-transform of the envelope")

    #Fit of the gaussians
    vals            = np.transpose([envelope[peaks], pulseTime[peaks,0]] )
    gaussianArray   = lambda x, *args: np.sum([vals[i,0]*np.exp(-(x-vals[i,1])**2/(2*el**2) ) for i, el in enumerate(args)], axis=0)

    if len(peaks) < 0:
        #Fit multifige gaussians into the envelope
        initialGuess    = np.full(len(vals), 10*fs_conv)
        popt,pcov   = optimize.curve_fit(gaussianArray, time, envelope, p0=initialGuess)
        #print(popt)
        axT.plot(time, gaussianArray(time, *popt), "g-", label="multifige Gauss-fit")

    axF.set_yscale("log")
    axT.set_yscale("log")
    axT.set_ylim([1e-3, 1])
    axT.legend()
    axF.legend()

    pl.show()
    pl.close()

def gaussianSum(x, A1, m1, s1, A2, m2, s2, A3, m3, s3):
    return A1*np.exp(-(x-m1)**2/(2*s1**2) )+A2*np.exp(-(x-m2)**2/(2*s2**2) )+A3*np.exp(-(x-m3)**2/(2*s3**2) )

def pulseFit(x, A, w1, phi, w2, sigma):
    return A*np.cos(w1*x+phi)*np.cos(w2*x)*np.exp(-x**2/(2*sigma**2) )

def besselFit(x, A, w1, phi, w2, sigma):
    return A*np.cos(w1*x+phi)*special.jv(0, w2*x)*np.exp(-np.abs(x)/sigma)



#Appendix to use main function
if __name__ == "__main__":
    main()
