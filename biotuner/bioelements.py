import numpy as np

def Angstrom_to_hertz(wavelength_in_Angstrom):
    c = 2.998e+8 # speed of light in m/s
    wavelength_in_meter = wavelength_in_Angstrom * 1e-10
    frequency = c / wavelength_in_meter
    return frequency


def nm_to_hertz(wavelength_in_nm):
    c = 2.998e+8 # speed of light in m/s
    wavelength_in_meter = wavelength_in_nm * 1e-9
    frequency = c / wavelength_in_meter
    return frequency


def hertz_to_nm(frequency_in_hertz):
    c = 2.998e+8 # speed of light in m/s
    wavelength_in_meter = c / frequency_in_hertz
    wavelength_in_nm = wavelength_in_meter * 1e9
    return wavelength_in_nm


def hertz_to_volt(frequency_in_hertz):
    h = 6.62607015e-34 # Planck's constant in J.s
    e = 1.60217662e-19 # electron charge in C
    voltage = h*frequency_in_hertz/e
    return voltage

def convert_to_nm(spectral_lines):
    converted_lines = {}
    for element, lines in spectral_lines.items():
        converted_lines[element] = [line * 10**9 for line in lines]
    return converted_lines

def find_matching_spectral_lines(peaks, non_metal, tolerance=1e-9):
    result = {}
    for peak in peaks:
        found = False
        n = 0
        scaled_peak = peak
        while not found:
            scaled_peak = scaled_peak / 2**n
            if scaled_peak < min(min(non_metal.values())):
                break
            for element, spectral_lines in non_metal.items():
                for line in spectral_lines:
                    if abs(scaled_peak - line) <= tolerance:
                        #print('LINE:', line, 'PEAK:', scaled_peak)
                        #print(scaled_peak-line)
                        if element not in result:
                            result[element] = []
                        result[element].append(scaled_peak)
                        found = True
                        break
                if found:
                    break
            n += 1
    return result

NON_METAL = {
    'hydrogen_balmer' : [
    656.28e-9, # H-alpha
    486.13e-9, # H-beta
    434.05e-9, # H-gamma
    410.17e-9, # H-delta
    397.01e-9, # H-epsilon
    388.90e-9, # H-zeta
    383.45e-9, # H-eta
    378.41e-9, # H-theta
    374.66e-9, # H-iota
    372.00e-9, # H-kappa
    369.47e-9, # H-lambda
    367.07e-9, # H-mu
    364.78e-9, # H-nu
    362.60e-9, # H-xi
    360.52e-9, # H-omicron
    358.54e-9, # H-pi
    356.66e-9, # H-rho
    354.87e-9, # H-sigma
    353.17e-9, # H-tau
    351.56e-9, # H-upsilon
    350.03e-9, # H-phi
    348.58e-9, # H-chi
    347.22e-9, # H-psi
    345.94e-9  # H-omega
], 'carbon':[
    1560.309e-9, # C II 1560
    1656.928e-9, # C IV 1548
    1907.0e-9, # C III 1907
    2325.e-9 # C III 2325
], 'nitrogen': [
    1485.0e-9, # N V 1240
    3995.0e-9, # N IV 7100
    4097.e-9, # N IV 7200
    4442.e-9, # N III 5200
    4447.e-9, # N III 5197
    4471.e-9, # N III 4640
    4607.e-9, # N III 4100
    4634.e-9, # N III 4103
    4640.e-9, # N III 4104
    4643.e-9, # N III 4103
    5199.e-9, # N III 3300
    5200.e-9, # N III 3300
    57.29e-9, # N II 5755
], 'oxygen' : [
    6300.e-9, # O I 6300
    6363.e-9, # O I 6363
    5577.e-9, # O I 5577
    6364.e-9, # O I 6364
    1661.e-9, # O III 1661
    1666.e-9, # O III 1666
    1671.e-9, # O III 1671
    2321.e-9, # O III 2321
    3345.e-9, # O III 3345
    4363.e-9, # O III 4363
    4931.e-9, # O III 4931
    5007.e-9, # O III 5007
    6158.e-9, # O III 6158
    6300.e-9, # O III 6300
    6364.e-9, # O III 6364
    6685.e-9, # O III 6685
    7319.e-9, # O III 7319
    7330.e-9, # O III 7330
], 'phosphorus': [1.43e-07, 1.439e-07, 1.449e-07, 1.459e-07, 1.469e-07,
                  1.478e-07, 1.511e-07, 1.529e-07, 1.538e-07, 1.548e-07,
                  1.557e-07, 1.566e-07, 1.575e-07, 1.585e-07, 1.594e-07,
                  1.599e-07, 1.611e-07, 1.62e-07, 1.63e-07, 1.64e-07,
                  1.65e-07, 1.658e-07, 1.668e-07, 1.677e-07, 1.687e-07,
                  1.696e-07, 1.706e-07, 1.715e-07, 1.725e-07, 1.734e-07,
                  1.743e-07, 1.753e-07, 1.762e-07, 1.772e-07, 1.781e-07,
                  1.8e-07, 1.81e-07, 1.819e-07, 1.829e-07, 1.838e-07,
                  1.847e-07, 1.857e-07, 1.866e-07, 1.875e-07, 1.885e-07,
                  1.894e-07, 1.9e-07],
'sulfur' : [    607.7e-09, 607.9e-09, 616.0e-09, 620.7e-09, 624.7e-09,
            633.5e-09, 635.4e-09, 636.4e-09, 638.5e-09, 643.8e-09,
            646.0e-09, 674.2e-09]
}

HALOGEN = {'fluorine': [810.2e-09, 811.0e-09, 815.4e-09, 817.4e-09,
                        819.2e-09, 824.0e-09, 825.2e-09, 827.3e-09,
                        835.3e-09, 837.9e-09],
           'chrlorine': [837.9e-09, 839.4e-09, 842.3e-09, 844.3e-09,
                         848.7e-09, 852.1e-09, 853.2e-09, 857.4e-09,
                         868.2e-09, 874.3e-09]
}
NOBLE_GAS = {'helium' : [
    587.5618e-9, # He I D3
    4471.47e-9, # He I 5876
    5015.68e-9, # He I 6678
    7065.22e-9, # He I 10830
    3964.73e-9, # He II 4686
    5411.52e-9 # He II 3203
], 'neon' : [
    1215.67e-9, # Ne VIII 780
    3426.53e-9, # Ne VI 3390
    3968.47e-9, # Ne V 3404
    5199.98e-9, # Ne V 3375
    5201.3e-9, # Ne V 3375
    5755.6e-9, # Ne III 6200
    6398.e-9, # Ne III 5200
    6402.e-9, # Ne III 5200
    6415.e-9, # Ne III 5200
    6424.e-9, # Ne III 5200
    6430.e-9, # Ne III 5200
    6435.e-9, # Ne III 5200
    6441.e-9, # Ne III 5200
    6443.e-9, # Ne III 5200
    6447.e-9, # Ne III 5200
    6462.e-9, # Ne III 5200
    6475.e-9, # Ne III 5200
    6491.e-9, # Ne III 5200
    6498.e-9, # Ne III 5200
    6506.e-9, # Ne III 5200
    6532.e-9, # Ne III 5200
], 
'argon' : [706.5e-09, 708.0e-09, 713.2e-09, 716.3e-09,
         718.8e-09, 728.1e-09, 730.1e-09, 731.0e-09,
         736.6e-09, 738.3e-09]

}


ALKA_METAL = {'lithium' : [
    6103.57e-9, # Li I 6104
    6708.78e-9, # Li I 6707
    8126.13e-9, # Li I 8126
    8883.99e-9, # Li I 8883
    9211.39e-9 # Li I 9211
], 'sodium' : [
    5894.6e-9, # Na I D1
    5895.9e-9, # Na I D2
    5688.2e-9, # Na I 
    5891.6e-9, # Na I 
    5896.1e-9, # Na I 
    8183.3e-9, # Na I
    8194.8e-9, # Na I
],
'potassium' : [    766.5e-09, 769.9e-09, 776.3e-09, 777.0e-09,     780.0e-09, 786.7e-09, 769.9e-09, 793.7e-09,     769.9e-09, 867.0e-09]

}

ALKA_EARTH_METAL = {'beryllium' : [
    3130.42e-9, # Be II 3130
    3526.53e-9, # Be II 3526
    4131.76e-9, # Be II 4131
    4226.73e-9 # Be II 4227
], 'magnesium' : [
    4481.1e-9, # Mg II 4481
    4571.1e-9, # Mg I 4571
    4575.4e-9, # Mg I 4575
    4702.0e-9, # Mg I 4702
    5528.4e-9, # Mg I 5528
    5711.1e-9, # Mg I 5711
    6318.7e-9, # Mg I 6318
    6319.2e-9, # Mg I 6319
    6319.5e-9, # Mg I 6319
    7167.4e-9, # Mg I 7167
],
'calcium' : [393.4e-09, 396.9e-09, 422.7e-09, 425.4e-09,
             431.2e-09, 433.1e-09, 436.3e-09, 486.1e-09,
             496.9e-09, 657.3e-09]

}

METALLOID = {'boron': [
    2497.07e-9, # B I 2497
    2501.13e-9, # B I 2501
    2508.40e-9, # B I 2508
    2511.99e-9, # B I 2512
    2524.17e-9, # B I 2524
    2528.02e-9, # B I 2528
    2532.11e-9, # B I 2532
    2537.58e-9, # B I 2538
    2542.60e-9 # B I 2543
],
    'silicon' : [1.41724e-07,1.265e-07,     1.2065e-07,     1.39376e-07,     1.53343e-07,     9.89870e-08]

}

ALL_ELEMENTS = {**NON_METAL, **HALOGEN, **NOBLE_GAS, **ALKA_METAL, **ALKA_EARTH_METAL, **METALLOID}
