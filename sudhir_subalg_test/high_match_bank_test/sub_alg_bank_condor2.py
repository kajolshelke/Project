#!/cvmfs/software.igwn.org/conda/envs/igwn-py39-lw/bin/python

import matplotlib.pyplot as pp
import pycbc.noise
import pycbc.psd
import pycbc.filter
from pycbc.filter import matched_filter
from pycbc.waveform import get_td_waveform
from pycbc.waveform import get_fd_waveform
import numpy as np
from pycbc.vetoes import power_chisq
from pycbc.events.ranking import newsnr
import pandas as pd
from pycbc.filter import sigma
import h5py
from pycbc.psd import interpolate, inverse_spectrum_truncation
import sys
import xml.etree.ElementTree as ET
from io import StringIO


# snr_desired1 = 20
# snr_desired2 = 15

buffer = 0.2

inj_time = 13

# bank = h5py.File('/Users/sudhirgholap/Documents/GW_lens4/H1L1-BANK2HDF-1126051217-1220400.hdf', 'r')
# Mass1, Mass2, Spin1z, Spin2z, F_lower = bank['mass1'][:], bank['mass2'][:], bank['spin1z'][:], bank['spin2z'][:], bank['f_lower'][:]

# mass1_bank, mass2_bank, chirpmass_bank, spin1z_bank, spin2z_bank, f_lower_bank  = [], [], [], [], [], []

# for (m1,m2, s1z, s2z, fl) in zip(Mass1, Mass2, Spin1z, Spin2z, F_lower):
#     if 9<= m1 <= 60 and  9<= m2 <= 60 and abs(s1z) <= 0.1 and abs(s2z) <= 0.1:
#         mass1_bank.append(m1)
#         mass2_bank.append(m2)
#         spin1z_bank.append(s1z)
#         spin2z_bank.append(s2z)

# print(len(mass1_bank))

# import xml.etree.ElementTree as ET
# from io import StringIO

# tree = ET.parse('testNonSpin99.xml')
# root = tree.getroot()


# f = StringIO(root[2][64].text[:-3])
# bank = np.loadtxt(f, delimiter=',',usecols=(40, 41) )

# # print((bank))
# mass1_bank, mass2_bank = [], []
# for i in range(len(bank)):
#     mass1_bank.append(bank[i][0]) 
#     mass2_bank.append(bank[i][1]) 

# print((bank))


# print(temp_bank)



def gen_template_bank(m_1, m_2, m_radius, temp_numb):

    # Generate templates for BNS
    temp_bank = np.zeros((temp_numb, 2))

    for i in range(temp_numb):
        r1 = np.random.uniform(0, m_radius)
        r2 = np.random.uniform(0, (np.pi*2))
        m1 = r1*(np.cos(r2)) + m_1
        m2 = r1*(np.sin(r2)) + m_2

        temp_bank[i, :] = [m1, m2]

    temp_bank[i,:] = [m_1, m_2]

    return temp_bank

def MF_bank(ts, bank, psd1, show_fig=False, xmin1=0, xmax1=1):
    # this function performs matched filter on a time series with a bank of templates and returns the list of SNR time series

    for i, _ in enumerate(bank):
        #print(f'Computing SNR for template {i+1}/{len(bank)}')
        # compute the SNR time series
        bank[i]['snr'], bank[i]['tpeak'], _, _, bank[i]['csnr_peak'] = gen_SNR(bank[i]['template'], ts, psd1)
        if show_fig:
            pp.figure(figsize=(15,3))
            abs(bank[i]['snr']).plot()
            pp.title(f"SNR for template {bank[i]['m1']:.1f},{bank[i]['m2']:.1f}")
            pp.xlim(xmin1, xmax1)
            pp.show()

    return bank

def best_trig(result):
    """
    Find the best trigger in a list of triggers.
    """
    maxSNR = 0
    for i, t in enumerate(result):
        if abs(t['csnr_peak']) > maxSNR:
            maxSNR = abs(t['csnr_peak'])
            maxIndex = i

    return maxIndex

def gen_noise(duration=64):

    # The color of the noise matches a PSD which you provide
    flow = 30.0
    delta_f = 1.0 / 16
    flen = int(2048 / delta_f) + 1
    psd = pycbc.psd.aLIGOZeroDetHighPower(flen, delta_f, flow)

    # Generate 256 seconds of noise at 4096 Hz
    delta_t = 1.0 / 4096
    tsamples = int(duration/ delta_t)
    ts = pycbc.noise.gaussian.noise_from_psd(tsamples, delta_t, psd, seed=127)

    return ts, psd

def preprocess_bank(bank, N, psd1=None, flow=30):

    new_bank = []
    for i, t in enumerate(bank):
        # print(t)
        h1 = gen_waveform('IMRPhenomPv2', t[0], t[1], 0, 0, 50, 0)

        h1.resize(N)
        h1 = h1.cyclic_time_shift(h1.sample_times[-1])
        sig1 = sigma(h1, psd=psd1, low_frequency_cutoff=flow)
        h1.data /= sig1
        # self_inner, _, _, _, _ = gen_SNR(h1, h1, psd1)

        new_bank.append({'m1': t[0], 'm2': t[1], 'template': h1})
        #print(f'preprocess bank loop {i+1}/{len(bank)}')
    return new_bank

def find_peak(snr, mask_between=None):

    if mask_between is not None:
        mask = (snr.sample_times >= mask_between[0]) & (snr.sample_times < mask_between[1])
        snr1 = snr.copy()
        snr1.data[mask] = 0
    else:
        snr1 = snr

    peak = abs(snr1).numpy().argmax()
    snrp = abs(snr1[peak])
    csnrp = snr1[peak]
    tpeak = snr1.sample_times[peak]

    return tpeak, peak, snrp, csnrp

def gen_SNR(template, data, psd):

    snr = matched_filter(template, data,
                         psd=psd, low_frequency_cutoff=30)

    tpeak, peak, snrp, csnrp = find_peak(snr)

    return snr, tpeak, peak, snrp, csnrp

def gen_waveform(model, m1, m2, s1z, s2z, distance, time):

    from pycbc.detector import Detector
    from pycbc.waveform import get_td_waveform

    ra = 1.7
    dec = 1.7
    pol = 0.2
    inclination_1 = 0
    d = Detector("H1")

    fp, fc = d.antenna_pattern(ra, dec, pol, time)
    hp_1, hc_1 = get_td_waveform(approximant=model,
                             mass1=m1, mass2=m2, spin1z=s1z, spin2z=s2z,
                             distance=distance, inclination=inclination_1,
                             delta_t=1.0/4096, f_lower=30)

    ht_1 = fp * hp_1 + fc * hc_1

    return ht_1

def overlap(ht_1,ht_2,buffer):

    if len(ht_1) < len(ht_2):
        ht_small, ht_big = ht_1, ht_2
    else:
        ht_small, ht_big = ht_2, ht_1

    dt = ht_small.delta_t

    # make buffer int multiple of dt
    buffer = np.round(buffer*ht_big.sample_rate)/ht_big.sample_rate
    # elongate the long template to accomodate enough space on left
    ht_big.resize(len(ht_big)+int(buffer*ht_big.sample_rate))
    # match small template
    ht_small.resize(len(ht_big))
    # correct the position after resizing
    ht_big = ht_big.cyclic_time_shift(buffer)
    #Shifting the merger time
    ht_small = ht_small.cyclic_time_shift(ht_small.start_time-ht_big.start_time-buffer)
    #Equating the start time of both signals
    ht_small.start_time = ht_big.start_time
    # resample to original sample rate
    ht_big = ht_big.resample(dt)
    ht_small = ht_small.resample(dt)

    #Combining the signals
    ht_total = ht_small + ht_big

    return ht_total, ht_2

def inject(ht_total,ts,injtime):

    # append extra zeros than required
    ht_total.append_zeros(int((ts.duration-injtime)*1.5*4096))

    ht_total = ht_total.cyclic_time_shift(int(injtime + ht_total.start_time))
    ht_total = ht_total.resample(ts.delta_t)

    #Resizing the signal
    ht_total.resize(len(ts))

    #Equating the start time
    ht_total = ht_total.cyclic_time_shift(ht_total.start_time + injtime)
    ht_total.start_time = ts.start_time

    #Injecting signal into noise
    ts = ts.add_into(ht_total)

    return ts

def shift_timeseries(tsx, tau):

    if 'float' in str(tsx.dtype):
        tsx_shifted = tsx.cyclic_time_shift(tau-tsx.duration)

    elif 'complex' in str(tsx.dtype):
        tsx_shifted_r = tsx.real().cyclic_time_shift(tau-tsx.duration)
        tsx_shifted_i = tsx.imag().cyclic_time_shift(tau-tsx.duration)
        tsx_shifted = tsx_shifted_r + 1j * tsx_shifted_i

    tsx_shifted.start_time = 0

    return tsx_shifted

def gen_template(model,m1,m2,s1z,s2z,conditioned):
    
    from pycbc.detector import Detector
    from pycbc.waveform import get_td_waveform

    ra = 1.7
    dec = 1.7
    pol = 0.2
    inclination_1 = 0
    time=0.0
    d = Detector("H1")

    # We get back the fp and fc antenna pattern weights.
    fp, fc = d.antenna_pattern(ra, dec, pol, time)

    hp, hc = get_td_waveform(approximant=model,
                     mass1=m1,
                     mass2=m2,spin1z=s1z,spin2z=s2z,
                     delta_t=conditioned.delta_t,
                     f_lower=30)

    ht_template = fp * hp + fc * hc
    ht_template.resize(len(conditioned))
    #Time shift 
    template = ht_template.cyclic_time_shift(ht_template.start_time)

    return template

def match_other_best_template_simple(itrig1, bank, psd, width=0.15, show_figs=False):
    rho_N = bank[itrig1]['snr']
    h1_bns = bank[itrig1]['template']
    tpeak = bank[itrig1]['tpeak']
    snr_peak_N = abs(bank[itrig1]['csnr_peak'])
    xmin, xmax = tpeak-0.5, tpeak+0.5
    max_snr = 0
    buffer = 0

    for i, t in enumerate(bank):

        # print(f"({t['m1']}, {t['m2']})")
        rho_B = t['snr']
        h1_bbh = t['template']

        # if show_figs:
        #     pp.figure(figsize=(15, 3))
        #     abs(rho_B).plot()
        #     abs(rho_N).plot()
        #     pp.xlim(xmin, xmax)
            # pp.show()
        #print(f'Computing inner product for template {i+1}/{len(bank)}')
        x_nb, _, _, _, _ = gen_SNR(h1_bbh, h1_bns, psd)
        shifted_nb = shift_timeseries(x_nb, tpeak)

        rho_sub = rho_B - snr_peak_N * shifted_nb

        # chisq_vals = gen_chisquare(h1_bbh, ?, t["m1"], t["m2"], 0, 0, psd)
        # new_snr = newsnr(abs(rho_sub), chisq_vals)

        tpeak2, _, snrp, csnrp = find_peak(rho_sub) #, mask_between=(tpeak-width, tpeak+width)

        bank[i]['csnr_peak'] = csnrp
        bank[i]['tpeak'] = tpeak2

        if snrp > max_snr:
            itrig2 = i
            max_snr = snrp
            tpeak2_best = tpeak2
            shifted_nb_best = shifted_nb
            rho_sub_best = rho_sub
            buffer_best = buffer

            if show_figs:
                print(f"({t['m1']}, {t['m2']}) {snrp} {tpeak2} {buffer}")
                pp.figure(figsize=(15, 6))
                pp.subplot(311)
                pp.title(f"peak {snr_peak_N:.2f}")
                abs(rho_B).plot()
                # abs(rho_N).plot()
                abs(rho_sub).plot()
                # (rho_sub).plot()
                pp.xlim(xmin, xmax)
                pp.subplot(312)
                abs(snr_peak_N * shifted_nb).plot()
                pp.xlim(xmin, xmax)
                # pp.ylim(-0.3, 11)
                pp.subplot(313)
                abs(rho_sub).plot()
                pp.xlim(xmin, xmax)
                # pp.ylim(-0.3, 11)
                pp.show()
    # pp.plot(shifted_nb_best.sample_times, shifted_nb_best)
    # pp.show()

    # pp.plot(rho_sub_best.sample_times, rho_sub_best)
    # pp.show()

    return itrig2, max_snr, tpeak2_best, buffer_best, bank

mass1_inj1_arr, mass2_inj1_arr, mass1_inj2_arr, mass2_inj2_arr, snr1_arr, snr2_arr = np.loadtxt(fname='injection_sub_alg_new.txt', delimiter=',', unpack='true')
# print(type(mass1_inj1_arr))
# print((snr1_arr[100]))
mass1_inj1, mass2_inj1, mass1_inj2, mass2_inj2, snr1, snr2 = [], [], [], [], [], []
for (m1_1,m2_1, m1_2, m2_2) in zip(mass1_inj1_arr, mass2_inj1_arr, mass1_inj2_arr, mass2_inj2_arr):
    if 12<= m1_1 <= 60 and  12<= m2_1 <= 60 and 12<= m1_2 <= 60 and  12<= m2_2 <= 60:
        mass1_inj1.append(m1_1)
        mass2_inj1.append(m2_1)
        mass1_inj2.append(m1_2)
        mass2_inj2.append(m2_2)

# print(f'length of mass1_inj1 {len(mass1_inj1)}')
mass1_inj1_arr = mass1_inj1[:900]
mass2_inj1_arr = mass2_inj1[:900]
mass1_inj2_arr = mass1_inj2[:900]
mass2_inj2_arr = mass2_inj2[:900]

# print(len(mass1_inj1_arr))

flow_global = 30
noise, psd = gen_noise(duration=16)
# print(noise.duration)
psd = interpolate(psd, noise.delta_f)
psd = inverse_spectrum_truncation(psd, int(4 * noise.sample_rate),
                                  low_frequency_cutoff=flow_global)

def gen_signal_for_injection(m1_inj, m2_inj, s1z_inj, s2z_inj, inc_inj, dist_inj):
    h_plus, h_cross = get_td_waveform(approximant='IMRPhenomPv2', 
                                            mass1= m1_inj, 
                                            mass2= m2_inj, 
                                            spin1z= s1z_inj,
                                            spin2z= s2z_inj,
                                            inclination= inc_inj,
                                            distance= dist_inj, 
                                            f_lower= flow_global, 
                                            delta_t=noise.delta_t)
    return h_plus, h_cross


rank = int(sys.argv[1])
m = 1

def get_optimal_snr(waveform, psd):

    waveform_to_use = waveform.copy()
    opt_snr = pycbc.filter.matchedfilter.overlap(waveform_to_use, waveform_to_use, psd=psd, low_frequency_cutoff=flow_global, normalized=False)**0.5
    return opt_snr

# print(len(snr1_arr))
for i in range(m*rank, m*(rank+1)):

    m1_1 = mass1_inj1_arr[i]
    m2_1 = mass2_inj1_arr[i]
    m1_2 = mass1_inj2_arr[i]
    m2_2 = mass2_inj2_arr[i]
    snr_desired1 = snr1_arr[i]
    snr_desired2 = snr2_arr[i]
    # print(m1_1, m2_1, m1_2, m2_2, snr_desired1, snr_desired2)
    h1_bbh = gen_waveform('IMRPhenomPv2', m1_1, m2_1, 0, 0, 1000, 0)
    h1_bns = gen_waveform('IMRPhenomPv2', m1_2, m2_2, 0, 0, 1000, 0)
    h1_bbh.resize(len(noise))
    h1_bns.resize(len(noise))
    # pp.plot(h1_bbh.sample_times, h1_bbh)
    # pp.show()

    factor1 = snr_desired1/get_optimal_snr(h1_bbh, psd)
    #print(f'factor1: {factor1}, opt_snr1: {get_optimal_snr(h1_bbh, psd)}')
    factor2 = snr_desired2/get_optimal_snr(h1_bns, psd)
    #print(f'factor2: {factor2}, opt_snr1: {get_optimal_snr(h1_bns, psd)}')

    ht_total, _ = overlap((h1_bbh.copy()*factor1), (h1_bns.copy()*factor2), buffer)
    ts1 = inject(ht_total, noise, injtime = inj_time)

    # pp.plot(ht_total.sample_times, ht_total)
    # pp.show()
    # pp.plot(noise.sample_times, noise)
    # pp.show()


    tree = ET.parse('testNonSpin999.xml')
    root = tree.getroot()


    f = StringIO(root[2][64].text[:-3])
    bank_file = np.loadtxt(f, delimiter=',',usecols=(40, 41) )

    # print((bank_file))
    mass1_bank, mass2_bank = [], []
    for i in range(len(bank_file)):
        mass1_bank.append(bank_file[i][0]) 
        mass2_bank.append(bank_file[i][1]) 
    # print(len(mass1_bank))

    # idx99_arr= []
    # for (m1, m2, i) in zip(mass1_arr, mass2_arr, range(len(mass1_arr))):
    #     hp2, _ = gen_signal_for_injection(m1, m2, 0, 0, 0.5, 1000)
    #     hp2.resize(len(noise))
    #     ip_bbh = pycbc.filter.matchedfilter.optimized_match(h1_bbh, hp2, psd=psd, low_frequency_cutoff=flow_global, return_phase=True)[0]
    #     ip_bns = pycbc.filter.matchedfilter.optimized_match(h1_bns, hp2, psd=psd, low_frequency_cutoff=flow_global, return_phase=True)[0]
    #     # ip_arr.append(ip)
    #     if max(ip_bbh, ip_bns) >= 0.999:
    #         idx99_arr.append(i)
    #         print(f'match loop {i + 1}/{len(mass1_arr)} {ip_bbh} {ip_bns}')

    # mass1_bank, mass2_bank = [], []

    # for idx in idx99_arr:
    #     mass1_bank.append(mass1_arr[idx])
    #     mass2_bank.append(mass2_arr[idx])

    temp_bank = np.zeros((len(mass1_bank), 2))

    for i in range(len(mass1_bank)):
        temp_bank[i, :] = [mass1_bank[i], mass2_bank[i]]

    '''Uncomment the following to test algorithm for In-situ bank'''  

    # R_bank = 0.2
    # N_bank = 400

    # np.random.seed(42)
    # temp_bank = np.append(gen_template_bank(m1_1, m2_1, R_bank, N_bank//2), 
    #                 gen_template_bank(m1_2, m2_2, R_bank, N_bank//2), 
    #                 axis=0)

    #print('bank preprocessing')
    bank = preprocess_bank(temp_bank, len(ts1), psd1=psd)
    #print('match_filtering...')
    result1 = MF_bank(ts1, bank, psd, show_fig=False, xmin1=35.5, xmax1=36.5)
    itrigger1 = best_trig(result1)

    result = {'trig':[],'index':[],'snr_rec':[], 'snr1':[], 'snr2':[], 'time':[], 'masses_inj':[], 'recovery':[]}   
    result["trig"].append('trig1')
    # result["best_trig1"].append(best_trig1)
    # result["best_trig2"].append(best_trig2)
    result["index"].append(itrigger1)
    result["masses_inj"].append((m1_1, m2_1, m1_2, m2_2))
    result["recovery"].append((result1[itrigger1]['m1'], result1[itrigger1]['m2']))
    # result["best_recovery"].append((mass1_bank[best_trig1], mass2_bank[best_trig1],mass1_bank[best_trig2], mass2_bank[best_trig2]))
    result["snr_rec"].append(abs(result1[itrigger1]['csnr_peak']))
    result['snr1'].append(snr_desired1)
    result['snr2'].append(snr_desired2)
    result["time"].append(result1[itrigger1]['tpeak'])

    FIGS = False
    for i in range(5):
        # choose best 2nd trigger
        #print(f'loop{i} 1st step')
        itrigger2, max_snr, tpeak2_best, buffer_best, result2 = match_other_best_template_simple(itrigger1, result1, psd, show_figs=FIGS)
        # print("trig2: ", itrigger2, result2[itrigger2]['m1'], result2[itrigger2]['m2'], abs(result2[itrigger2]['csnr_peak']), result2[itrigger2]['tpeak'])

        result["trig"].append('trig2')
        # result["best_trig1"].append(best_trig1)
        # result["best_trig2"].append(best_trig2)
        result["index"].append(itrigger2)
        result["masses_inj"].append((m1_1, m2_1, m1_2, m2_2))
        result["recovery"].append((result2[itrigger2]['m1'], result2[itrigger2]['m2']))
        # result["best_recovery"].append((mass1_bank[best_trig1], mass2_bank[best_trig1],mass1_bank[best_trig2], mass2_bank[best_trig2]))
        result["snr_rec"].append(abs(result2[itrigger2]['csnr_peak']))
        result['snr1'].append(snr_desired1)
        result['snr2'].append(snr_desired2)
        result["time"].append(result2[itrigger2]['tpeak'])


        # adjust 1st trigger
        #print(f'loop{i} 2nd step')
        itrigger1, max_snr, tpeak2_best, buffer_best, result1 = match_other_best_template_simple(itrigger2, result2, psd, show_figs=FIGS)
        # print("\t\t\ttrig1: ", itrigger1, result1[itrigger1]['m1'], result1[itrigger1]['m2'], abs(result1[itrigger1]['csnr_peak']), result1[itrigger1]['tpeak'])

        result["trig"].append('trig1')
        # result["best_trig1"].append(best_trig1)
        # result["best_trig2"].append(best_trig2)
        result["index"].append(itrigger1)
        result["masses_inj"].append((m1_1, m2_1, m1_2, m2_2))
        result["recovery"].append((result1[itrigger1]['m1'], result1[itrigger1]['m2']))
        # result["best_recovery"].append((mass1_bank[best_trig1], mass2_bank[best_trig1],mass1_bank[best_trig2], mass2_bank[best_trig2]))
        result["snr_rec"].append(abs(result1[itrigger1]['csnr_peak']))
        result['snr1'].append(snr_desired1)
        result['snr2'].append(snr_desired2)
        result["time"].append(result1[itrigger1]['tpeak'])

        data = pd.DataFrame.from_dict(result)
        data.to_csv('sub_alg_%d.csv'%rank)
