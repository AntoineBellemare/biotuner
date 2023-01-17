import numpy as np

c = 299792458
visible_range_THz = [380, 750]
visible_range_Hz = [3.8*10**14, 7.5*10**14]
visible_range_nm = [750, 380]


def wavelength_to_rgb(wavelength, gamma=0.5):

    '''This converts a given wavelength of light to an 
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    '''

    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        #R = ((-(wavelength - 440) / (440 - 380))) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        #R = (1.0) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    R *= 255
    G *= 255
    B *= 255
    return (int(R), int(G), int(B))

def scale2freqs (scale, fund, THz = True):
    if THz == True:
        scale = [Hz2THz(s) for s in scale]
    scale_freqs = []
    for s in scale:
        scale_freqs.append(s*fund)
    return scale_freqs
def nm2Hz (nm, c):
    nm = (c/nm)*10**9
def Hz2nm (Hz, c):
    nm = (c/Hz)*10**9
    return nm
def Hz2THz (freq):
    freq = freq/10**12
    return freq

def THz2Hz (freq):
    freq = freq*10**12
    return freq

def audible2visible (freq, visible_range = visible_range_Hz, c = 299792458):
    i = 0
    new_freq = 0
    while new_freq < visible_range_Hz[0]:
        i+=1
        octave = 2**i
        new_freq = freq*octave
        #print(new_freq)
    n_octave = i
    Hz = new_freq
    THz = Hz2THz(new_freq)
    nm = Hz2nm(Hz, c)
    
    return THz, Hz, nm, n_octave


def wavelength_to_frequency(wavelengths, min_frequency, max_frequency):
    c = 2.998 * 10**17 # speed of light in nm/s
    frequencies = c / np.array(wavelengths)
    n = 0
    while(np.min(frequencies) < min_frequency or np.max(frequencies) > max_frequency):
        frequencies = frequencies / 2
        n += 1
    return frequencies, n


def viz_scale_colors(scale, fund, title=None):
    # set default title
    if title == None:
        title = 'Color palette derived from biological tuning'
    min_ = 0
    max_ = 1
    # convert the scale to frequency values
    scale_freqs = scale2freqs(scale, fund)
    # compute the averaged consonance of each step
    scale_cons, _ = tuning_cons_matrix(scale, dyad_similarity, ratio_type='all')
    # rescale to match RGB standards (0, 255)
    scale_cons = ((np.array(scale_cons) - min_) * (1/max_ - min_) * 255).astype('uint8')
    img_array = []
    for s, cons in zip(scale_freqs, scale_cons):
        # convert freq in nanometer values
        _, _, nm, octave = audible2visible(s)
        # convert to RGB values
        rgb = wavelength_to_rgb(nm)
        # convert to HSV values
        hsv = colorsys.rgb_to_hsv(rgb[0]/float(255),rgb[1]/float(255), rgb[2]/float(255))
        hsv = np.array(hsv)
        # rescale
        hsv = ((hsv - 0) * (1/(1 - 0) * 255)).astype('uint8')
        hsv = list(hsv)
        # define the saturation
        hsv[1] = int(cons)
        # define the luminance
        hsv[2] = 200
        hsv = tuple(hsv)
        img = Image.new('HSV', (300, 300), hsv)
        img_array.append(img)

    # Figure parameters
    fig = plt.figure(figsize=(16., 16.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(2, int(len(scale_freqs)/2)),  # creates 2x2 grid of axes
                    axes_pad=0.1,  # pad between axes in inch.
                    )
    for ax, im in zip(grid, img_array):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
    plt.show()