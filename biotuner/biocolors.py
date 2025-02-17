import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.colors import hsv_to_rgb
import colorsys
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
from biotuner.metrics import tuning_cons_matrix, dyad_similarity

c = 299792458
visible_range_THz = [380, 750]
visible_range_Hz = [3.8*10**14, 7.5*10**14]
visible_range_nm = [750, 380]


def wavelength_to_rgb(wavelength, gamma=0.5):

    """
    Convert a given wavelength of light to an approximate RGB color value.
    
    The input wavelength must be given in nanometers (nm) in the range from
    380 nm through 750 nm (789 THz through 400 THz). The function is based on
    code by Dan Bruton: http://www.physics.sfasu.edu/astro/color/spectra.html
    
    Parameters
    ----------
    wavelength : float
        The wavelength of light in nanometers.
    gamma : float, optional
        The gamma correction factor (default is 0.5).
        
    Returns
    -------
    tuple
        A tuple containing the RGB color values (R, G, B) as integers in the range 0-255.
    
    Examples
    --------
    >>> wavelength_to_rgb(475)
    (0, 213, 255)
    
    >>> wavelength_to_rgb(650, gamma=1.0)
    (246, 0, 0)
    """

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

def scale2freqs (scale, fund, THz=True):
    """
    Returns a list of frequency values given a scale and fundamental frequency.

    Parameters
    ----------
    scale : list
        A list of values representing the scale to be used.
    fund : float
        The fundamental frequency to be used for scaling the scale.
    THz : bool, optional
        Whether the input scale needs to be transformed from Hz to THz. Defaults to True.

    Returns
    -------
    list
        A list of frequency values obtained by scaling the input scale with the fundamental frequency.

    """

    freqs = [x*fund for x in scale]
    if THz == True:
        freqs = [Hz2THz(s) for s in freqs]
    return freqs

def nm2Hz (nm, c):
    """
    Converts a wavelength value from nm to Hz.

    Parameters
    ----------
    nm : float
        The wavelength value in nm to be converted.
    c : float
        The speed of light in nm/s.

    Returns
    -------
    float
        The frequency value in Hz.

    """
    Hz = (c/nm)*10**9
    return Hz

def Hz2nm (Hz, c):
    """
    Converts a frequency value from Hz to nm.

    Parameters
    ----------
    Hz : float
        The frequency value in Hz to be converted.
    c : float
        The speed of light in nm/s.

    Returns
    -------
    float
        The wavelength value in nm.

    """
    nm = (c/Hz)*10**9
    return nm

def Hz2THz (freq):
    """
    Converts a frequency value from Hz to THz.

    Parameters
    ----------
    freq : float
        The frequency value in Hz to be converted.

    Returns
    -------
    float
        The frequency value in THz.

    """
    freq = freq/10**12
    return freq

def THz2Hz (freq):
    """
    Converts a frequency value from THz to Hz.

    Parameters
    ----------
    freq : float
        The frequency value in THz to be converted.

    Returns
    -------
    float
        The frequency value in Hz.

    """
    freq = freq*10**12
    return freq

def audible2visible (freq, visible_range = visible_range_Hz, c = 299792458):
    """
    Converts an audible frequency value to its corresponding visible frequency value.

    Parameters
    ----------
    freq : float
        The audible frequency value to be converted.
    visible_range : tuple, optional
        A tuple of two values representing the lower and upper bounds of the visible frequency range in Hz. Defaults to visible_range_Hz (380-750 THz).
    c : float, optional
        The speed of light in nm/s. Defaults to 299792458.

    Returns
    -------
    tuple
        A tuple containing the frequency value in THz, Hz, nm, and the number of octaves shifted to obtain a frequency within the visible range.

    """
    i = 0
    new_freq = 0
    while new_freq < visible_range_Hz[0]:
        i+=1
        octave = 2**i
        new_freq = freq*octave
    n_octave = i
    Hz = new_freq
    THz = Hz2THz(new_freq)
    nm = Hz2nm(Hz, c)
    
    return THz, Hz, nm, n_octave

def wavelength_to_frequency(wavelengths, min_frequency, max_frequency):
    """
    Converts a list of wavelength values to their corresponding frequency values, scaling them to fit within a given frequency range.

    Parameters
    ----------
    wavelengths : list
        A list of wavelength values to be converted.
    min_frequency : float
        The lower bound of the desired frequency range.
    max_frequency : float
        The upper bound of the desired frequency range.

    Returns
    -------
    tuple
        A tuple containing the frequency values obtained by converting the input wavelengths and
        the number of times the frequencies were halved to fit within the desired frequency range.

    """
    c = 2.998 * 10**17 # speed of light in nm/s
    frequencies = c / np.array(wavelengths)
    n = 0
    while(np.min(frequencies) < min_frequency or np.max(frequencies) > max_frequency):
        frequencies = frequencies / 2
        n += 1
    return frequencies, n

def viz_scale_colors(scale, fund, title=None, return_fig=False):
    """
    Visualize a color palette derived from biological tuning by converting
    the input musical scale to frequency values and mapping them to RGB colors.

    The hue is based on the frequency values and the saturation based on the
    average consonance of the scale step with all other steps.

    Parameters
    ----------
    scale : list
        List of scale values representing the musical scale.
    fund : float
        The fundamental frequency value.
    title : str, optional
        Title for the visualization (default is 'Color palette derived from biological tuning').

    Returns
    -------
    None
        Displays the color palette derived from the input scale as an image grid.

    Examples
    --------
    >>> scale = [1, 9/8, 5/4, 4/3, 3/2, 5/3, 15/8, 2]
    >>> viz_scale_colors(scale, fund=30)
    """
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
        # define the saturation based on consonance
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
    if return_fig:
        return fig
    else:
        plt.show()
    
def animate_colors(colors, duration, frames_per_second, filename='test'):
    """
    Animate a sequence of colors and save the animation as a GIF file.
    
    The input colors should be in the HSV color space. The function generates
    a smooth animation by interpolating between the input colors and updates
    the rectangular patch color in each frame.

    Parameters
    ----------
    colors : list of tuples
        List of HSV colors represented as tuples (H, S, V).
    duration : float
        Duration of the animation in seconds.
    frames_per_second : int
        Number of frames per second for the animation.
    filename : str, optional
        Filename for the output GIF file (default is 'test').

    Returns
    -------
    None
        Saves the animation as a GIF file with the specified filename.

    Examples
    --------
    >>> colors = [(0.1, 1, 1), (0.5, 1, 1), (0.9, 1, 1)]
    >>> animate_colors(colors, 5, 30, 'color_animation')
    """
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Set axis limits
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Create a rectangular patch
    rect = plt.Rectangle((0, 0), 1, 1, color=hsv_to_rgb(colors[0]))
    ax.add_patch(rect)

    def update(frame):
        color_index = int(frame // (frames_per_second * duration / len(colors)))
        next_color_index = (color_index + 1) % len(colors)
        color_weight = frame % (frames_per_second * duration / len(colors)) / (frames_per_second * duration / len(colors))
        color = hsv_to_rgb((colors[color_index][0] + (colors[next_color_index][0] - colors[color_index][0]) * color_weight,
                            colors[color_index][1] + (colors[next_color_index][1] - colors[color_index][1]) * color_weight,
                            colors[color_index][2] + (colors[next_color_index][2] - colors[color_index][2]) * color_weight))
        rect.set_color(color)

    # Create animation using the update function
    ani = FuncAnimation(fig, update, frames=np.linspace(0, frames_per_second * duration - 1, frames_per_second * duration), repeat=True)
    #plt.show()
    # embedding for the video
    #html = display.HTML(video)
    
    # draw the animation
    #display.display(html)
    #plt.close()
    ani.save('{}.gif'.format(filename))
