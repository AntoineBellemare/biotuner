import streamlit as st
from biotuner.biotuner_object import compute_biotuner
from biotuner.biotuner_utils import create_SCL, segment_time_series, rebound, create_midi
from biotuner.scale_construction import tuning_reduction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from streamlit_echarts import st_echarts
import librosa
import librosa.display
from pytuning.visualizations.scales import consonance_matrix
from biotuner.metrics import dyad_similarity
from biotuner.biotuner_utils import scale2frac
from fractions import Fraction
import sounddevice as sd
import time
import os
import io
import plotly.graph_objects as go
from music21 import stream, chord, note
import base64
import colorsys
import matplotlib.colors as mcolors

import wave
import tempfile
from music21 import stream, chord, note

import json
from biotuner.biocolors import audible2visible, scale2freqs, wavelength_to_rgb
from biotuner.biotuner_object import dyad_similarity
from biotuner.metrics import tuning_cons_matrix
import webcolors

from struct import pack
import json

import random




def generate_chord_wave(tuning, base_freq=440, duration=1.0, sample_rate=44100):
    """
    Generate a waveform for a random chord from the given tuning.

    Parameters:
    - tuning: List of frequency ratios.
    - base_freq: The fundamental frequency.
    - duration: Duration of the chord in seconds.
    - sample_rate: Sampling rate for audio playback.

    Returns:
    - numpy array containing the waveform of the chord.
    """
    num_notes = random.randint(3, 4)  # Select 3 to 4 notes for the chord
    selected_ratios = random.sample(list(tuning), num_notes)  # Choose random tuning ratios
    freqs = [base_freq * float(Fraction(str(ratio))) for ratio in selected_ratios]

    t = np.linspace(0, duration, int(sample_rate * duration), False)
    chord_wave = sum(0.3 * np.sin(2 * np.pi * f * t) for f in freqs)  # Sum waveforms
    chord_wave /= max(abs(chord_wave))  # Normalize to prevent clipping
    # reduce the volume again
    chord_wave *= 0.5
    # add a fade in and fade out
    fade_duration = 1
    fade_samples = int(sample_rate * fade_duration)
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    chord_wave[:fade_samples] *= fade_in
    chord_wave[-fade_samples:] *= fade_out


    return chord_wave, freqs


def play_random_chords(tuning, num_chords=3, base_freq=440, duration=1.0, sample_rate=44100, silence_duration=0.2):
    """
    Play a sequence of random chords.

    Parameters:
    - tuning: List of frequency ratios.
    - num_chords: Number of chords to play.
    - base_freq: The fundamental frequency.
    - duration: Duration of each chord.
    - sample_rate: Sampling rate.
    - silence_duration: Pause between chords.
    """
    for _ in range(num_chords):
        chord_wave, freqs = generate_chord_wave(tuning, base_freq, duration, sample_rate)
        print(f"🎵 Playing Chord with frequencies: {[round(f, 2) for f in freqs]} Hz")
        sd.play(chord_wave, samplerate=sample_rate)
        time.sleep(duration + silence_duration)
    sd.stop()


def save_random_chords(tuning, num_chords=3, base_freq=440, duration=1.0, sample_rate=44100, silence_duration=0.2):
    """
    Generate and save a sequence of random chords as a .wav file.

    Parameters:
    - tuning: List of frequency ratios.
    - num_chords: Number of chords to generate.
    - base_freq: The fundamental frequency.
    - duration: Duration of each chord.
    - sample_rate: Sampling rate.
    - silence_duration: Pause between chords.
    
    Returns:
    - Path to the saved .wav file.
    """
    all_waves = []
    silence = np.zeros(int(sample_rate * silence_duration))  # Silence between chords

    for _ in range(num_chords):
        chord_wave, _ = generate_chord_wave(tuning, base_freq, duration, sample_rate)
        all_waves.append(chord_wave)
        all_waves.append(silence)  # Add silence after each chord

    final_wave = np.concatenate(all_waves)  # Combine all chords with silence

    # Save as a temporary .wav file
    temp_wav_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(temp_wav_file.name, "w") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes((final_wave * 32767).astype(np.int16).tobytes())  # Convert to 16-bit PCM

    print(f"✅ Chords saved to {temp_wav_file.name}")
    return temp_wav_file.name  # Return the file path

def export_ase(colors, filename="palette.ase"):
    """
    Export color palette to ASE (Adobe Swatch Exchange) format.
    
    :param colors: Dict of color names and their hex values.
    :param filename: Name of the output ASE file.
    """
    with open(filename, "wb") as f:
        f.write(b"ASEF")  # ASE file signature
        f.write(pack(">I", 1))  # Version (1)
        f.write(pack(">I", len(colors)))  # Number of colors

        for name, hex_color in colors.items():
            rgb = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (1, 3, 5))
            name_encoded = name.encode("utf-16be") + b"\x00\x00"  # Proper UTF-16BE encoding

            f.write(pack(">H", 0xC001))  # Start block
            f.write(pack(">H", len(name_encoded) // 2))  # Name length in 2-byte words
            f.write(name_encoded)  # Name
            f.write(b"RGB ")  # Color mode
            f.write(pack(">fff", *rgb))  # RGB values
            f.write(pack(">H", 2))  # Color type (global)

    print(f"✅ ASE file saved as {filename}")



def export_json(colors, filename="palette.json"):
    """
    Export color palette to JSON format.
    
    :param colors: Dict of color names and hex values.
    :param filename: Output JSON filename.
    """
    data = {"colors": [{"name": name, "hex": hex_val} for name, hex_val in colors.items()]}
    with open(filename, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"✅ JSON file saved as {filename}")

def export_svg(colors, filename="palette.svg"):
    """
    Export color palette to an SVG file.
    
    :param colors: Dict of color names and hex values.
    :param filename: Output SVG filename.
    """
    svg_template = """<?xml version="1.0" encoding="UTF-8" standalone="no"?>
    <svg width="500" height="100" xmlns="http://www.w3.org/2000/svg">
        {}
    </svg>"""

    rectangles = []
    x_offset = 0
    width = 500 // max(1, len(colors))  # Ensure no division by zero

    for name, hex_color in colors.items():
        rect = f'''
        <rect x="{x_offset}" y="0" width="{width}" height="100" fill="{hex_color}" stroke="black" stroke-width="2">
            <title>{name} ({hex_color})</title>
        </rect>'''
        rectangles.append(rect)
        x_offset += width

    svg_content = svg_template.format("\n".join(rectangles))

    with open(filename, "w", encoding="utf-8") as f:
        f.write(svg_content)

    print(f"✅ SVG file saved as {filename}")



def export_css(colors, filename="palette.css"):
    """
    Export color palette to a CSS file.
    
    :param colors: Dict of color names and hex values.
    :param filename: Output CSS filename.
    """
    css_content = "/* Generated Color Palette */\n:root {\n"
    for name, hex_color in colors.items():
        css_var = name.lower().replace(" ", "-").replace("_", "-")  # Sanitize names
        css_content += f"    --color-{css_var}: {hex_color};\n"
    css_content += "}"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(css_content)

    print(f"✅ CSS file saved as {filename}")


def export_gpl(colors, filename="palette.gpl"):
    """
    Export color palette to a GPL (GIMP Palette) file.
    
    :param colors: Dict of color names and hex values.
    :param filename: Output GPL filename.
    """
    gpl_content = "GIMP Palette\n"
    gpl_content += "Name:CustomPalette\n"
    gpl_content += "#\n"  # Required separator

    for name, hex_color in colors.items():
        r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (1, 3, 5))
        gpl_content += f"{r} {g} {b} {name}\n"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(gpl_content)

    print(f"✅ GPL file saved as {filename}")



def tuning_to_colors(tuning, fundamental=440):
    """Convert a tuning scale to a color palette using HSV transformation from Biotuner."""

    min_ = 0
    max_ = 1

    # Convert tuning to frequency values
    scale_freqs = scale2freqs(tuning, fundamental)

    # Compute consonance scores (normalized between 0 and 1)
    scale_cons, _, _ = tuning_cons_matrix(tuning, dyad_similarity, ratio_type="all")
    scale_cons = (np.array(scale_cons) - min_) * (1 / max_ - min_) * 255
    scale_cons = scale_cons.astype("uint8").astype(float) / 255

    hsv_all = []
    color_names = []
    hex_colors = []

    for freq, cons in zip(scale_freqs, scale_cons):
        _, _, nm, octave = audible2visible(freq)  # Convert frequency to visible light spectrum
        rgb = wavelength_to_rgb(nm)  # Convert to RGB
        hsv = colorsys.rgb_to_hsv(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255)

        # Adjust HSV values
        hsv = np.array(hsv)
        hsv[1] = cons  # Set saturation to consonance
        hsv[2] = 200 / 255  # Fixed brightness

        # Convert to RGB and HEX
        rgb_tuple = tuple(int(c * 255) for c in colorsys.hsv_to_rgb(*hsv))
        hex_value = "#{:02x}{:02x}{:02x}".format(*rgb_tuple)
        
        # Get color name
        color_name = rgb2name_matplotlib(rgb_tuple)

        hsv_all.append(hsv)
        hex_colors.append(hex_value)
        color_names.append(color_name)

    return hsv_all, hex_colors, color_names



def rgb2name_matplotlib(rgb):
    """
    Convert an RGB tuple to the closest named color in Matplotlib's color dictionary.

    Parameters:
        rgb (Tuple[int, int, int]): RGB color tuple

    Returns:
        str: Closest color name from Matplotlib's color dictionary.
    """
    rgb_normalized = tuple(v / 255 for v in rgb)  # Normalize to 0-1
    min_distance = float("inf")
    closest_color = None

    for name, hex_value in mcolors.CSS4_COLORS.items():
        color_rgb = mcolors.hex2color(hex_value)  # Convert to (0-1) range
        distance = sum((c1 - c2) ** 2 for c1, c2 in zip(rgb_normalized, color_rgb))

        if distance < min_distance:
            min_distance = distance
            closest_color = name

    return closest_color

def musicxml_to_base64(xml_bytes):
    """ Convert MusicXML file bytes to base64 for embedding in HTML. """
    return base64.b64encode(xml_bytes.getvalue()).decode("utf-8")

def chords_to_musicxml(chords, bound_times, fname="chord_progression", total_duration=None):
    """
    Convert extracted chords into a MusicXML file with valid durations.
    
    Parameters:
    - chords: list of chord frequencies
    - bound_times: list of time stamps for each chord (in sample indices)
    - fname: filename for saving the MusicXML file
    
    Returns:
    - xml_bytes: BytesIO object containing the MusicXML data
    """

    score = stream.Score()
    part = stream.Part()

    # Convert bound times to durations
    if len(bound_times) < 2:
        raise ValueError("Not enough bound times to determine durations.")

    durations = [bound_times[i+1] - bound_times[i] for i in range(len(bound_times) - 1)]

    # Make sure `durations` matches the number of chords
    durations = durations[:len(chords)]  # Trim durations to match chords


    if total_duration:
        # Scale durations to match total_duration
        scale_factor = total_duration / sum(durations)
        durations = [d * scale_factor for d in durations]
    else:
        # Ensure durations retain natural variation and map to reasonable note lengths
        max_duration = max(durations)  # Get the longest segment duration
        durations = [max(0.25, (d / max_duration) * 4) for d in durations]  # Scale so max duration is ~4 beats



    # Standard durations for music notation
    standard_durations = [4.0, 2.0, 1.0, 0.5, 0.25, 0.125]  # Whole, Half, Quarter, Eighth, Sixteenth
    for i, chord_freqs in enumerate(chords[1:]):
        if len(chord_freqs) == 0:
            continue  # Skip empty chords
        
        # Convert frequencies to closest MIDI notes
        # Apply octave shift: multiply frequencies by 2^(n_oct_up)
        chord_freqs = [rebound(x, low=np.min(chord_freqs)-1, high=np.min(chord_freqs)*2, octave=2) for x in chord_freqs]
        n_oct_up = st.session_state.n_oct_up  # Retrieve octave shift value
        shift_factor = 2 ** n_oct_up
        shifted_freqs = [f * shift_factor for f in chord_freqs]

        
        # Convert shifted frequencies to MIDI
        midi_pitches = [round(69 + 12 * np.log2(f / 440.0)) for f in shifted_freqs]


        # Create a chord object
        music_chord = chord.Chord(midi_pitches)
        # Find the closest standard duration
        duration = max(0.25, durations[i] * 4)  # Ensure minimum duration
        closest_duration = min(standard_durations, key=lambda x: abs(x - duration))  # Snap to valid duration

        music_chord.duration.quarterLength = closest_duration  # Assign duration

        part.append(music_chord)

    score.append(part)

    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".musicxml") as tmpfile:
        temp_filename = tmpfile.name
        score.write(fmt="musicxml", fp=temp_filename)  # Save to file

    # Read back into BytesIO
    with open(temp_filename, "rb") as f:
        xml_bytes = io.BytesIO(f.read())

    xml_bytes.seek(0)  # Reset buffer position

    return xml_bytes




import numpy as np
import librosa


def generate_segments(data, n_segments=64, sf=44100, time_resolution=10, frequency_resolution=1000):
    """
    Extracts spectral centroid features and segments audio.

    Parameters:
    - data: np.array, audio signal
    - n_segments: int, number of segments for agglomerative clustering
    - sf: int, sample rate (Hz)
    - time_resolution: int, frame step in milliseconds (controls hop_length)
    - frequency_resolution: int, FFT window size in samples (controls n_fft)

    Returns:
    - segments: list of segmented arrays (excluding very short ones)
    - bound_samples: array of segment boundary indices
    """

    # Convert high-level parameters to librosa parameters
    n_fft = frequency_resolution  # Number of samples per FFT
    hop_length = max(1, int((time_resolution / 1000) * sf))  # Convert ms to samples

    print(f'Using frequency_resolution={n_fft} samples, time_resolution={time_resolution} ms → hop_length={hop_length} samples')

    # Compute spectral centroid feature
    feature = librosa.feature.spectral_centroid(y=data, sr=sf, n_fft=n_fft, hop_length=hop_length)
    #feature = librosa.feature.spectral_flatness(y=data, n_fft=n_fft, hop_length=hop_length)

    # Check feature size before segmentation
    n_features = feature.shape[1]  # Number of time steps in feature extraction
    print("Feature shape:", feature.shape, "Number of features:", n_features, 'Requested segments:', n_segments)

    try:
        # Segment using agglomerative clustering
        bounds = librosa.segment.agglomerative(feature, n_segments)

        # Convert from frames to time
        bound_times = librosa.frames_to_time(bounds, sr=sf)
        bound_times = np.clip(bound_times, 0, len(data) / sf)  # Ensure within bounds
        bound_samples = (bound_times * sf).astype(int)

        # Generate segments
        segments = segment_time_series(data, bound_samples)
        segments = [segment[~np.isnan(segment)] for segment in segments]

        # Filter out segments that are too short (<2 samples)
        valid_segments = [seg for seg in segments if len(seg) >= 2]
        num_valid_segments = len(valid_segments)

        # Check if enough valid segments are generated
        if num_valid_segments < 0.25 * n_segments:
            st.warning(f"⚠️ Only {num_valid_segments} valid segments generated, fewer than requested {n_segments}.")
            st.write("""
                🔧 **Try adjusting the following parameters:**
                - Increase the **length of the signal** (more than 2 seconds)
                - Increase the **time resolution** (e.g., from 10ms to 50ms)
                - Increase the **FFT window size** (e.g., from 1000 to 2048)
                - Reduce the **number of segments** (e.g., from 64 to a lower value)
            """)
            return segments, bound_samples

        return segments, bound_samples

    except ValueError as e:
        st.error(f"🚨 Segmentation Error: {e}")
        st.write("""
            🔧 **Try adjusting the following parameters:**
            - Incase the **length of the signal** (more than 2 seconds)
            - Increase the **time resolution** (e.g., from 10ms to 50ms)
            - Increase the **FFT window size** (e.g., from 1000 to 2048)
            - Reduce the **number of segments** (e.g., from 64 to a lower value)
        """)
        return [], []




def extract_chords(segments, method='cepstrum', n_peaks=5, peaks_idxs=None, precision=0.01, max_freq=100, sf=44100):
    """
    Extract chords from segments using spectral peaks detection.
    
    Parameters
    ----------
    segments : list of 1D arrays
        List of segments to extract chords from.
    method : str
        Method to use for peaks detection.
        Options are 'cepstrum' and 'EMD'. Default is 'cepstrum'.
    n_peaks : int
        Number of peaks to extract. Default is 5.
    peaks_idxs : list of ints
        List of indexes of peaks to extract when using EMD method. Default is None.
    precision : float
        Precision of the peaks detection. Default is 0.01.
    max_freq : float
        Maximum frequency for peaks detection. Default is 100.
        
    Returns
    -------
    chords : list of lists
        List of extracted chords.
    """
    # remove nans from segments
    chords = []
    n_segments_removed = 0
    for segment in segments[1:]:
        #print('segment size:', segment.size)
        if segment.size < sf/2:
            n_segments_removed += 1
            continue
        if np.isnan(segment).any():
            segment = segment[~np.isnan(segment)]
        
        #normalize segment between 0 and 1
        segment = (segment - np.nanmin(segment)) / (np.nanmax(segment) - np.nanmin(segment))
        # print('LENGTH OF SEGMENT:', len(segment))
        try:
            bt = compute_biotuner(sf=sf, data=segment, peaks_function=method, precision=precision)
            bt.peaks_extraction(min_freq=0.001, max_freq=max_freq, n_peaks=n_peaks, nIMFs=n_peaks)
        except Exception as e:
            print('Error:', e)
            n_segments_removed += 1
            continue
        print('PEAKS', bt.peaks)

        chord=[]
        if peaks_idxs is not None:
            for i in peaks_idxs:
                if i < len(bt.peaks):
                    chord.append(bt.peaks[i])
        else:
            chord = bt.peaks[1:]
        chords.append(chord)
    return chords, n_segments_removed

def chords_to_MIDI(chords, bound_times, n_oct_up=7, max_beat_duration=1.5, 
                   total_duration=None, foldername='test', fname='test_chords', microtonal=True):
    """
    Convert chords to MIDI file while controlling total duration accurately.
    
    Parameters
    ----------
    chords : list
        List of chords to convert to MIDI.
    bound_times : list
        List of time stamps for each chord (in sample indices).
    n_oct_up : int
        Number of octaves to shift the chords up. Default is 7.
    max_beat_duration : float
        Maximum beat duration. Default is 1.5.
    total_duration : float
        Desired total duration of the MIDI file in seconds (optional).
    foldername : str
        Name of the folder to save the MIDI file.
    fname : str
        Name of the MIDI file.
    microtonal : bool
        If True, use microtonal MIDI.

    Returns
    -------
    midi_file : MIDI file
        MIDI file of the chords.
    """
    n_chords = len(chords)
    mult = 2**n_oct_up
    new_chords = []

    for chord in chords[1:n_chords]:
        c = [rebound(x, low=np.min(chord)-1, high=np.min(chord)*2, octave=2) for x in chord]
        c = [int(x*mult) for x in c]
        new_chords.append(c)

    # Convert bound times to seconds
    bound_times_sec = bound_times / bound_times[-1] * total_duration if total_duration else bound_times / 1000

    # Compute durations in seconds
    durations = [bound_times_sec[i+1] - bound_times_sec[i] for i in range(n_chords-1)]

    # Normalize durations between 0.1 and max_beat_duration
    durations = [d / np.max(durations) for d in durations]
    durations = [d * max_beat_duration for d in durations]

    # Ensure total duration matches requested value
    if total_duration:
        total_duration = total_duration*2
        scale_factor = total_duration / sum(durations)
        durations = [d * scale_factor for d in durations]

    # Debugging prints
    print(f"Requested Total Duration: {total_duration}s")
    print(f"Computed Total Duration: {sum(durations)}s (should match requested)")
    print(f"Durations after rescaling: {durations}")

    if not os.path.exists(f'../plant_music/spectral_chords/{foldername}'):
        os.makedirs(f'../plant_music/spectral_chords/{foldername}')
    fname = f"../plant_music/spectral_chords/{foldername}/{fname}"
    
    midi_file = create_midi(new_chords, durations, subdivision=1, microtonal=microtonal, filename=fname)
    return midi_file


# Initialize session state for storing uploaded data
if "uploaded_data" not in st.session_state:
    st.session_state.uploaded_data = None
if "sampling_rate" not in st.session_state:
    st.session_state.sampling_rate = None
if "num_steps" not in st.session_state:
    st.session_state.num_steps = 7
if "selection" not in st.session_state:
    st.session_state.selection = None
if "start_time" not in st.session_state:
    st.session_state.start_time = 0.0
if "end_time" not in st.session_state:
    st.session_state.end_time = None  # Will be set dynamically


# Apply Dark Theme Styling
st.markdown(
    """
    <style>
        body { background-color: #121212; color: #f5deb3; }
        .stButton>button { 
            background-color: #FFB6C1; 
            color: white; 
            border-radius: 10px; 
            padding: 12px 24px; 
            font-size: 18px; 
            transition: background-color 0.3s ease-in-out;
        }
        .stButton>button:hover { 
            background-color: #FFB6C1 !important;  /* Soft Pink */
            color: black !important;
        }
        .stTextInput>div>div>input, .stFileUploader>div>div>div { background-color: #1f1f1f; color: #f5deb3; }
        .stTabs [data-baseweb="tab"] {
            font-size: 64px !important; /* Adjust size */
            font-weight: bold !important; /* Make it bold */
            padding: 20px !important; /* Increase padding */
        }
        .big-title { font-size: 26px !important; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown(
    """
    <style>
    div[data-testid="stNumberInput"] { 
        position: relative; 
        z-index: 999;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <style>
        /* Increase the font size of tab names */
        div[data-baseweb="tab"] {
            font-size: 24px !important;  /* Adjust font size */
            font-weight: bold !important;  /* Make it bold */
            padding: 14px 24px !important; /* Increase padding */
        }
    </style>
    """,
    unsafe_allow_html=True
)


# --- Sidebar Configuration ---

st.sidebar.title("⚙️ Biotuning Settings")

# Peak Extraction Methods
st.sidebar.subheader("Peak Extraction")
peak_extraction_method = st.sidebar.radio(
    "Select a method:", 
    ["Harmonic Recurrence","Intermodulation Components", "EMD", "EEG Bands", 'FOOOF'],
)

st.session_state.peak_method = peak_extraction_method
method_mapping = {
    "EMD": "EMD",
    "EEG Bands": "fixed",
    "Harmonic Recurrence": "harmonic_recurrence",
    "Intermodulation Components": "EIMC",
    "FOOOF": "FOOOF",
}
# Precision Setting
precision = st.sidebar.select_slider(
    "Precision (Hz)",
    options=[0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10],
    value=1
)

# Sampling Frequency Input
st.sidebar.subheader("Sampling Frequency")
sampling_frequency = st.sidebar.number_input(
    "Enter the sampling frequency (Hz):", 
    min_value=1, 
    value=st.session_state.sampling_rate if st.session_state.sampling_rate != None else 256,
    step=1
)

# Number of Peaks Input
st.sidebar.subheader("Number of Peaks")
n_peaks = st.sidebar.number_input(
    "Enter the number of peaks:", 
    min_value=1, 
    value=5, 
    step=1
)

# --- Main Interface ---
col1, col2 = st.columns([1, 5])
with col1:
    st.image("assets/biotuner_logo.png", width=150)
with col2:
    st.markdown(
        """
        <style>
        .custom-title {
            margin-bottom: 0px; /* Adjust spacing between title and subtitle */
        }
        .custom-subtitle {
            margin-top: 0px;
            margin-bottom: 0px;
        }
        </style>
        <h1 class="custom-title">Biotuner Engine</h1>
        <h3 class="custom-subtitle"><i>🎼 Harmonic Analysis of Time Series</i></h3>
        """,
        unsafe_allow_html=True
    )


#st.markdown("Upload your **audio or data file** to analyze its **harmonic properties**.")


# --- CSS for Styling ---
st.markdown("""
    <style>
        .signal-container {
            display: flex;
            justify-content: space-around;
            align-items: center;
            padding: 15px;
            background-color: #1f1f1f;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        .signal-box {
            text-align: center;
            color: #f5deb3;
            font-size: 16px;
            width: 120px;
            cursor: pointer;
        }
        .signal-box:hover {
            transform: scale(1.1);
            transition: 0.3s ease-in-out;
        }
        .selected {
            background-color: rgba(255, 255, 255, 0.1);
            padding: 5px;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)


# --- CSS Styling ---
st.markdown("""
    <style>
        .signal-container {
            display: flex;
            justify-content: space-around;
            align-items: center;
            padding: 15px;
            background-color: #1f1f1f;
            border-radius: 10px;
            margin-bottom: 10px;
        }
        .signal-box {
            text-align: center;
            color: #f5deb3;
            font-size: 20px;
            width: 150px;
            cursor: pointer;
            padding: 15px;
            border-radius: 10px;
            transition: 0.3s ease-in-out;
        }
        .signal-box:hover {
            transform: scale(1.1);
        }
        .selected {
            background-color: rgba(255, 255, 255, 0.1);
            padding: 5px;
            border-radius: 10px;
        }
        .stButton>button {
            font-size: 24px !important;
            height: 80px !important;
            width: 160px !important;
            border-radius: 12px !important;
            border: 2px solid transparent;
            transition: 0.3s;
        }
        .stButton>button:hover {
            transform: scale(1.1);
        }
        .stButton>button:focus {
            border: 2px solid white;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state for selected signal
if "selected_signal" not in st.session_state:
    st.session_state.selected_signal = None

def show_info(signal_type):
    info_dict = {
        "Audio": "<b>🔊 Audio Signals</b><br><br>"
                 "Audio signals are everywhere. Try to find the tuning of a bell, the resonance of a yawn, or the subharmonies of a whale song. "
                 "Transform the takeoff of a plane into a chord, or extract the hidden harmonies of the river where you swim.<br><br>"
                 "<b>Resources:</b><br>"

                 "🔹 <a href='https://www.sonicvisualiser.org/' target='_blank'>Sonic Visualiser</a> – Advanced tool for visualizing and analyzing audio features.<br>"
                 "🔹 <a href='https://freesound.org/' target='_blank'>Freesound</a> – A collaborative database of audio samples, field recordings, and sound effects.<br>"
                 "🔹 <a href='https://librosa.org/' target='_blank'>Librosa</a> – Python library for audio analysis and feature extraction.",

        "Brain (EEG)": "<b>🧠 Brain Signals (EEG)</b><br><br>"
                       "EEG captures electrical activity from populations of neurons in the brain, measured via electrodes on the scalp. "
                       "It provides insights into neural oscillations, event-related potentials (ERP), and functional connectivity. "
                       "<b>Resources:</b><br>"
                       "🔹 <a href='https://openneuro.org/' target='_blank'>OpenNeuro</a> – A large repository of open EEG, fMRI, and MEG datasets for neuroscience research.<br>"
                       "🔹 <a href='http://bnci-horizon-2020.eu/database' target='_blank'>BNCI Horizon 2020</a> – A collection of EEG datasets related to brain-computer interfaces (BCI) and cognitive experiments.<br>"
                       "🔹 <a href='https://openbci.com/' target='_blank'>OpenBCI</a> – Open-source EEG hardware and software platform for neurophysiology research and BCI applications.",

        "Heart (ECG)": "<b>❤️ Heart Signals (ECG)</b><br><br>"
                       "ECG measures the heart's electrical activity, revealing information on cardiac cycles, heart rate variability (HRV), and arrhythmias. "
                       "Got a smartwatch? Use your own heart rate data to explore the rhythm and music of your pulses.<br>"
                       "<b>Resources:</b><br>"
                       "🔹 <a href='https://physionet.org/' target='_blank'>PhysioNet</a> – Open database of ECG recordings.<br>"
                       "🔹 <a href='https://neuropsychology.github.io/NeuroKit/' target='_blank'>Neurokit2</a> – A Python toolbox for physiological signal processing.<br>",

        "Smartphone Sensors": "<b>📱 Smartphone Sensors</b><br><br>"
                              "Smartphones integrate multiple sensors including accelerometers (motion), gyroscopes (rotation), "
                              "magnetometers (orientation), and PPG (heart rate). These sensors enable multimodal physiological and behavioral tracking.<br><br>"
                              "<b>Apps for Data Recording:</b><br>"
                              "🔹 <a href='https://www.vieyrasoftware.net/' target='_blank'>Physics Toolbox Sensor Suite</a> – Logs multiple smartphone sensor streams.<br>"
                             "🔹 <a href='https://hexler.net/touchosc' target='_blank'>TouchOSC</a> – A powerful mobile app for sending sensor data via OSC, widely used for real-time interactive applications.<br>"
                              "🔹 <a href='https://phyphox.org/' target='_blank'>Phyphox</a> – Open-source app for experimental sensor data collection.",

        "Plant": "<b>🌿 Plant Signals</b><br><br>"
                 "Plants generate bioelectrical signals influenced by external stimuli such as light, touch, and environmental changes. "
                 "These signals can be measured as action potentials, impedance changes, or ion channel activity. "
                 "Let's shift our timeframes to listen to the music of the plants.<br><br>"
                 "<b>Resources:</b><br>"
                 "🔹 <a href='https://backyardbrains.com/products/PlantSpikerBox' target='_blank'>Plant SpikerBox</a> – DIY plant electrophysiology kit.<br>",

        "Your Creativity": "<b>🎨 Your Creativity</b><br><br>"
                   "Any data that fluctuates over time can become a sound, a visualization, or an insight. "
                   "Try transforming stock market, dance movements or even gravitational waves into harmonies.<br><br> "
                   
    }
    return info_dict.get(signal_type, "No information available.")


# --- Clickable Signal Icons ---
st.markdown("### 📡 Supported Modalities")
col1, col2, col3, col4, col5, col6 = st.columns(6)

# Button styling based on selection
button_colors = {
    "Audio": "#ff6b6b",  # Red
    "Brain (EEG)": "#6b6bff",  # Blue
    "Heart (ECG)": "#ff4d4d",  # Deep Red
    "Smartphone Sensors": "#ffb86b",  # Orange
    "Plant": "#6bff6b",  # Green
    "Your Creativity": "#8A2BE2",  # Violet
}

# Function to set selected signal
def select_signal(signal):
    st.session_state.selected_signal = signal

# Display buttons as larger, colorful icons
with col1:
    if st.button("🎵 Audio", key="audio", help="Click to learn about Audio Signals"):
        select_signal("Audio")
with col2:
    if st.button("🧠 Brain (EEG)", key="brain", help="Click to learn about EEG Signals"):
        select_signal("Brain (EEG)")
with col3:
    if st.button("❤️ Heart (ECG)", key="heart", help="Click to learn about ECG Signals"):
        select_signal("Heart (ECG)")
with col4:
    if st.button("📱 Smartphone Sensors", key="smartphone", help="Click to learn about Smartphone Sensors"):
        select_signal("Smartphone Sensors")
with col5:
    if st.button("🌿 Plant", key="plant", help="Click to learn about Plant Bioelectrical Signals"):
        select_signal("Plant")
with col6:
    if st.button("🎨 Your Creativity", key="creativity", help="Click to explore your own creative data"):
        select_signal("Your Creativity")

# Display information panel dynamically
if st.session_state.selected_signal:
    st.markdown(
        f'<div style="border-left: 8px solid {button_colors[st.session_state.selected_signal]}; padding: 10px; background-color: rgba(255,255,255,0.1); border-radius: 10px;">'
        
        f'{show_info(st.session_state.selected_signal)}</div>',
        unsafe_allow_html=True
    )


# # Display information panel dynamically
# if st.session_state.selected_signal:
#     st.info(show_info(st.session_state.selected_signal))

# # Display information when a signal is selected
# show_info(st.session_state.selected_signal)
# File Upload
st.markdown(
    """
    <hr style="border: 3px solid white; margin-top: 20px; margin-bottom: 20px;">
    """, unsafe_allow_html=True
)


uploaded_file = st.file_uploader("Upload your **audio or data file**", type=["wav", "mp3", "csv"])

if uploaded_file:
    st.success("✅ File uploaded successfully!")
    
    # --- Process WAV File ---
    if uploaded_file.name.endswith('.wav') or uploaded_file.name.endswith('.mp3'):
        #st.write("🔊 Processing audio file:", uploaded_file.name)
        
        # Load audio
        y, sr = librosa.load(uploaded_file, sr=None)
        st.session_state.uploaded_data = y  # Store original data
        print('Data uploaded:', len(st.session_state.uploaded_data))
        st.session_state.sampling_rate = sr  # Store sampling rate
        # update the value of the sampling frequency in the sidebar
        if st.session_state.peak_method == "FOOOF":
            st.session_state.peak_method = "Harmonic Recurrence"
        #st.rerun()

    # --- Process CSV File ---
    elif uploaded_file.name.endswith('.csv'):
        st.write("📊 Processing CSV file:", uploaded_file.name)
        

        st.session_state.csv_data = pd.read_csv(uploaded_file)
        
        # Let user choose column, and store it persistently
        column_to_use = st.selectbox("Select column for time series:", st.session_state.csv_data.columns)

        if column_to_use:  # Only proceed if a column is selected
            st.session_state.uploaded_data = st.session_state.csv_data[column_to_use].values
            st.session_state.sampling_rate = sampling_frequency  # Assume user-defined

    else:
        st.write("Unsupported file format. Please upload a .wav or .csv file.")

    # Time array
    time_axis = np.linspace(0, len(st.session_state.uploaded_data) / st.session_state.sampling_rate, num=len(st.session_state.uploaded_data))

    # Default end time is the full length of the signal
    if st.session_state.end_time is None:
        st.session_state.end_time = time_axis[-1]

    # UI for manual selection
    # st.write("🔍 Select the time range to highlight:")
    col1, col2 = st.columns(2)
    start_manual = col1.number_input("Start Time (s)", min_value=0.0, value=st.session_state.start_time, step=0.1, format="%.2f")
    end_manual = col2.number_input("End Time (s)", min_value=0.0, value=st.session_state.end_time, step=0.1, format="%.2f")

    # Confirm Selection Button

    # Store the new selection in session state
    st.session_state.start_time = start_manual
    st.session_state.end_time = end_manual

    # Extract selected portion
    selected_indices = (time_axis >= st.session_state.start_time) & (time_axis <= st.session_state.end_time)
    selected_signal = st.session_state.uploaded_data[selected_indices]

    

    #st.success(f"✅ Selection Updated: {st.session_state.start_time:.2f}s - {st.session_state.end_time:.2f}s")

    # Create Plotly figure with highlighted selection
    fig = go.Figure()

    # Add the full waveform
    fig.add_trace(go.Scatter(x=time_axis, y=st.session_state.uploaded_data, mode="lines", name="Audio Signal", line=dict(color="deepskyblue")))

    # Add highlighted selected region
    selected_indices = (time_axis >= st.session_state.start_time) & (time_axis <= st.session_state.end_time)
    fig.add_trace(go.Scatter(
        x=time_axis[selected_indices], 
        y=selected_signal, 
        mode="lines", 
        name="Selected Region", 
        line=dict(color="red", width=3)
    ))

    # Update layout
    fig.update_layout(
        title="Interactive Waveform Selection",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        height=400  # Set a fixed height
    )

    # Update `uploaded_data` with selected portion
    st.session_state.uploaded_data = np.copy(selected_signal)
    print('selected signal length:', len(st.session_state.uploaded_data))

    # Display Plotly chart
    st.plotly_chart(fig, use_container_width=True)
    #st.rerun()

    
        
        # # Keep the plot persistent
        # if "uploaded_data" in st.session_state and st.session_state.uploaded_data is not None:
        #     fig, ax = plt.subplots(figsize=(8, 4))
        #     fig.patch.set_alpha(0.0)
        #     ax.patch.set_alpha(0.0)
        #     ax.plot(st.session_state.uploaded_data, color='#40E0D0')
        #     ax.set_title("Time Series Plot", color='white')
        #     ax.set_xlabel("Index", color='white')
        #     ax.set_ylabel(column_to_use, color='white')
        #     ax.tick_params(axis='x', colors='white')
        #     ax.tick_params(axis='y', colors='white')
        #     ax.grid(False)
        #     st.pyplot(fig)


    

# --- Tuning Analysis ---



tab1, tab2, tab3 = st.tabs(["Tuning", "Chords", "Biocolors"])

with tab1:
    st.subheader("Tuning Analysis")

    tuning_method = st.selectbox(
        "Tuning Parameter", 
        ['Peaks Ratios', "Harmonic Fit", "Dissonance Curve"]
    )
    max_denom = st.number_input("Max Denominator", min_value=1, value=100, step=1)

    if st.session_state.uploaded_data is not None:
        if st.button("Run Tuning Analysis"):
            print('length of uploaded data:', len(st.session_state.uploaded_data))
            # Initialize Biotuner Object
            try:
                biotuning = compute_biotuner(
                    sf=st.session_state.sampling_rate, 
                    peaks_function=method_mapping[st.session_state.peak_method],
                    precision=precision, 
                    n_harm=10
                )

                # Extract spectral peaks
                biotuning.peaks_extraction(
                    st.session_state.uploaded_data, 
                    ratios_extension=True, 
                    max_freq=sampling_frequency / 2, 
                    n_peaks=n_peaks,
                    graph=False, 
                    min_harms=2
                )
            except:
                if st.session_state.peak_method in ["Harmonic Recurrence", "Intermodulation Components"]:
                    st.error(f"⚠️ No peaks detected. Try reducing the precision in Hz, or changing method.")
                    st.stop()  # Stop execution immediately after the warning
                if st.session_state.peak_method in ["EMD"]:
                    st.error("⚠️ No peaks detected for tuning analysis. Consider changing method.")
                    st.stop()
                
            if len(biotuning.peaks) < 2:
                if st.session_state.peak_method in ["EMD"]:
                    st.error("⚠️ Not enough peaks detected for tuning analysis. Consider changing method.")
                    st.stop()
                
            # Compute tuning
            if tuning_method == "Peaks Ratios":
                tuning = biotuning.peaks_ratios 
            elif tuning_method == "Dissonance Curve":
                biotuning.peaks_extension(
                    method='harmonic_fit', 
                    harm_function='mult',  
                    n_harm=20, 
                    cons_limit=0.05, 
                    ratios_extension=True, 
                    scale_cons_limit=0.1
                )
                biotuning.compute_diss_curve(
                    plot=False, 
                    input_type='extended_peaks', 
                    euler_comp=False, 
                    denom=max_denom, 
                    max_ratio=2, 
                    n_tet_grid=12
                )
                tuning = biotuning.diss_scale
            elif tuning_method == "Harmonic Fit":
                tuning = biotuning.harmonic_fit_tuning(n_harm=128, bounds=0.1, n_common_harms=50)

            


            # Round tuning & remove duplicates
            tuning = np.round(np.unique(tuning), 5)
            # Convert tuning to fractions with max denominator of 100
            tuning, _, _ = scale2frac(tuning, max_denom)
            # remove duplicates from the tuning
            tuning = np.unique(tuning)
            # Store tuning in session state
            st.session_state.tuning = tuning

            if len(tuning) < 2:
                st.error("⚠️ Not enough unique tuning ratios detected. Try adjusting the parameters.")
                st.stop()  # Stop execution immediately after the warning
            
            tuning_consonance = []
            for index1 in range(len(tuning)):
                for index2 in range(len(tuning)):
                    if tuning[index1] != tuning[index2]:
                        entry = tuning[index1] / tuning[index2]
                        tuning_consonance.append(dyad_similarity(entry))
            tuning_consonance = np.mean(tuning_consonance)
            st.session_state.tuning_consonance = tuning_consonance

    # --- Plot Tuning Analysis ---
    if "tuning" in st.session_state and st.session_state.tuning is not None:
        st.subheader("🎵 Tuning Analysis Results")

        # Create two columns: left for tuning list, right for the wheel
        col1, col2 = st.columns([1, 2])  # 1:1 ratio (adjust if needed)

        with col1:
            st.write("### Tuning Values")

            # Create DataFrame for structured display
            df_tuning = pd.DataFrame({
                "Ratio": st.session_state.tuning,
            })

            # Display as dataframe with width 200px
            st.dataframe(df_tuning, width=200)


        with col2:
            # Compute consonance metric
            tuning_consonance = np.round(st.session_state.tuning_consonance, 2)

            # Define gauge (wheel)
            wheel_options = {
                "series": [
                    {
                        "type": "gauge",
                        "detail": {"formatter": "{value}%"},
                        "data": [{"value": tuning_consonance, "name": "Tuning Consonance"}],
                        "axisLine": {
                            "lineStyle": {
                                "width": 12,
                                "color": [[0.3, "#ff5733"], [0.7, "#ffbd33"], [1, "#33ff57"]]
                            }
                        },
                        "min": 0,
                        "max": 50
                    }
                ]
            }
            st_echarts(wheel_options, height="400px")
        # Reduce space before the matrix
        # Reduce space before the consonance matrix using CSS
        st.markdown("""
            <style>
                div.block-container { padding-top: 0rem; }
                .stDataFrame { margin-bottom: -30px; }  /* Reduce gap between table and wheel */
                .stPlotlyChart { margin-top: -50px; }  /* Bring matrix closer */
            </style>
        """, unsafe_allow_html=True)

        # Keep consonance matrix below
        st.write("### Tuning Consonance Matrix")
        cons_matrix_harmsim = consonance_matrix(
            st.session_state.tuning, metric_function=dyad_similarity, vmin=0, vmax=50, cmap='magma', fig=None
        )
        st.pyplot(cons_matrix_harmsim)


    if "tuning" in st.session_state:
        
        st.subheader("🔊 Play Tuning")
        num_chords = st.slider("🎵 Number of Random Chords", min_value=1, max_value=10, value=3)
        st.session_state.num_chords = num_chords  # Store selection
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🎶 Play Random Chords"):
                play_random_chords(st.session_state.tuning, num_chords=st.session_state.num_chords, base_freq=300, duration=2)
            if st.button("💾 Save Random Chords as .wav"):
                wav_file_path = save_random_chords(st.session_state.tuning, num_chords=st.session_state.num_chords, base_freq=300, duration=2)
                with open(wav_file_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Chords as WAV",
                        data=f,
                        file_name="random_chords.wav",
                        mime="audio/wav"
                    )
                #play_random_chord(st.session_state.tuning, base_freq=300, duration=2)
        with col2:
            if st.button("Play Tuning"):
                def play_tuning(tuning, base_freq=120, duration=0.2, sample_rate=44100):
                    """
                    Generate and play sine waves corresponding to the tuning.
                    tuning: List of frequency ratios (relative to base frequency)
                    base_freq: Reference frequency (default: Middle C = 261.63 Hz)
                    duration: Note duration in seconds
                    sample_rate: Sampling rate for audio
                    """
                    for ratio in tuning:
                        freq = base_freq * ratio  # Convert ratio to Hz
                        t = np.linspace(0, duration, int(sample_rate * duration), False)
                        wave = 0.5 * np.sin(2 * np.pi * freq * t)  # Generate sine wave
                        # create a more complex tone using FM synthesis
                        modulator = 0.5 * np.sin(2 * np.pi * 2 * freq * t)
                        wave = 0.5 * np.sin(2 * np.pi * (freq + 5 * modulator) * t)

                        print(f"🎵 Playing {freq:.2f} Hz")  # Display the frequency
                        sd.play(wave, samplerate=sample_rate)
                        time.sleep(duration)  # Hold the note
                        sd.stop()
                
                tuning = [float(Fraction(str(x))) for x in st.session_state.tuning]
                play_tuning(tuning)
        with col3:
            if st.button("Save Tuning"):
                # add 2/1 to the tuning
                st.session_state.tuning = np.append(st.session_state.tuning, 2)
                scl_content = create_SCL(st.session_state.tuning, 'biotuning')
                st.download_button(
                    label="Download SCL File",
                    data=scl_content,
                    file_name="tuning.scl",
                    mime="text/plain"
                )

        st.subheader("Reduce Tuning")

        num_steps = st.slider("Select number of steps:", 2, len(st.session_state.tuning), st.session_state.num_steps)
        st.session_state.num_steps = num_steps  # Store selection

        if st.button("Reduce Tuning"):
            # Convert tuning to float before reduction
            tuning_floats = [float(Fraction(str(x))) for x in st.session_state.tuning]
            
            _, reduced_tuning, new_tuning_consonance = tuning_reduction(tuning_floats, num_steps, function=dyad_similarity)
            # order the reduced tuning
            reduced_tuning = np.round(np.unique(reduced_tuning), 5)
            reduced_tuning, _, _ = scale2frac(reduced_tuning, 100)
            st.session_state.reduced_tuning = reduced_tuning
            st.session_state.new_tuning_consonance = new_tuning_consonance

    

        # --- Plot Reduced Tuning ---
    # --- Plot Reduced Tuning ---
    # --- Plot Reduced Tuning ---
    if "reduced_tuning" in st.session_state:
        new_tuning_consonance = st.session_state.new_tuning_consonance
        st.subheader("🎵 Reduced Tuning Analysis")

        # Create two columns: left for reduced tuning list, right for the wheel
        col1, col2 = st.columns([1, 2])  # 1:1 ratio (adjust if needed)

        with col1:
            reduced_tuning = st.session_state.reduced_tuning
            reduced_tuning = np.unique(reduced_tuning)  # Ensure no duplicates


            # Create DataFrame for structured display
            df_reduced_tuning = pd.DataFrame({
                "Ratio": reduced_tuning
            })

            # Display as table (same as original tuning table)
            st.write("### Reduced Tuning Values")
            st.dataframe(df_reduced_tuning, width=200)  # Adjust width if needed

        with col2:
            # Compute new tuning consonance metric
            new_tuning_consonance = np.round(new_tuning_consonance, 2)

            # Define gauge (wheel)
            wheel_options = {
                "series": [
                    {
                        "type": "gauge",
                        "detail": {"formatter": "{value}%"},
                        "data": [{"value": new_tuning_consonance, "name": "Tuning Consonance"}],
                        "axisLine": {
                            "lineStyle": {
                                "width": 12,
                                "color": [[0.3, "#ff5733"], [0.7, "#ffbd33"], [1, "#33ff57"]]
                            }
                        },
                        "min": 0,
                        "max": 50
                    }
                ]
            }
            st_echarts(wheel_options, height="400px")

        st.markdown("""
            <style>
                div.block-container { padding-top: 0rem; }
                .stDataFrame { margin-bottom: -30px; }  /* Reduce gap between table and wheel */
                .stPlotlyChart { margin-top: -50px; }  /* Bring matrix closer */
            </style>
        """, unsafe_allow_html=True)
        # Keep consonance matrix below
        st.write("### Reduced Tuning Consonance Matrix")
        cons_matrix_reduced = consonance_matrix(
            st.session_state.reduced_tuning, metric_function=dyad_similarity, vmin=0, vmax=50, cmap='magma', fig=None
        )
        st.pyplot(cons_matrix_reduced)



    # --- Save Reduced Tuning Button ---
    if "reduced_tuning" in st.session_state:
        if st.button("Save Reduced Tuning"):
            # add 2/1 to the tuning
            st.session_state.reduced_tuning = np.append(st.session_state.reduced_tuning, 2)
            scl_content_reduced = create_SCL(st.session_state.reduced_tuning, 'biotuning_reduced')
            st.download_button(
                label="Download Reduced SCL File",
                data=scl_content_reduced,
                file_name="reduced_tuning.scl",
                mime="text/plain"
            )



# --- Chords Section ---
with tab2:
    data_ = st.session_state.uploaded_data
    st.subheader("Chords Analysis")

    microtonal = st.radio("Microtonal?", ["Yes", "No"])
    st.session_state.microtonal = microtonal  # Store selection
    n_segments = st.slider("Number of Segments", 1, 128, 16)
    st.session_state.n_segments = n_segments  # Store selection
    n_oct_up = st.slider("Number of Octaves Up", 0, 10, 5)
    st.session_state.n_oct_up = n_oct_up  # Store selection

    # Generate log-spaced values from 1 Hz to 10,000 Hz
    log_values = np.logspace(np.log10(1), np.log10(10000), num=50)
    log_values = np.round(log_values).astype(int)  # Convert to integers

    # Ensure 100 Hz is in the list (avoid ValueError)
    if 100 not in log_values:
        log_values = np.append(log_values, 100)
        log_values = np.unique(log_values)  # Keep unique sorted values

    # Streamlit Log-Space Slider
    max_freq = st.select_slider(
        "Max Frequency (Hz, Log Scale)", 
        options=log_values.tolist(),  # Convert to list for Streamlit
        value=100  # Ensure this value exists in `options`
    )
    st.session_state.max_freq = max_freq  # Store selection
    # Create three columns for input fields
    col1, col2, col3 = st.columns(3)

    with col1:
        frequency_resolution = st.number_input("Frequency Resolution (samples)", min_value=1, value=1000, step=1)
        st.session_state.frequency_resolution = frequency_resolution  # Store selection

    with col2:
        time_resolution = st.number_input("Temporal Resolution (ms)", min_value=1, value=10, step=1)

    with col3:
        total_duration = st.number_input("Total Duration (s)", min_value=1, value=30, step=1)


    if st.button("Generate Chords"):

        segments, bound_times = generate_segments(data_,
                                                  n_segments=st.session_state.n_segments, sf=st.session_state.sampling_rate,
                                                  time_resolution=time_resolution, frequency_resolution=frequency_resolution)
        print('Number of segments:', len(segments))
        print('Length of segments:', [len(s) for s in segments])
        # Plot the time series and the change points
        fig, ax = plt.subplots(figsize=(10, 5))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        ax.plot(st.session_state.uploaded_data, color='#40E0D0', label='Time Series')
        ax.vlines(bound_times, data_.min(), data_.max(), color='r', linestyle='--', label='Change Points')
        ax.set_xlabel('Time', color='white')
        ax.set_ylabel('Amplitude', color='white')
        ax.set_title('Time Series with Change Points', color='white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.legend(loc='upper right', frameon=False, fontsize='large', facecolor='none', edgecolor='none', labelcolor='white')
        # remove gridlines
        ax.grid(False)
        # Store the plot in session state
        st.session_state["chords_plot"] = fig
        

        chords, n_segments_removed = extract_chords(segments, method=method_mapping[peak_extraction_method],
                                n_peaks=n_peaks, peaks_idxs=None, precision=precision,
                                max_freq=st.session_state.max_freq, sf=st.session_state.sampling_rate) 
        st.write(f"Number of segments removed: {n_segments_removed}")
        if len(chords) <= 1:
            st.error("No chords extracted. Please adjust the parameters."
            "")
            st.stop()
        
        xml_bytes = chords_to_musicxml(chords, bound_times, total_duration=total_duration)

        print('CHORDS', chords, 'NUmber of chords:', len(chords))
        MIDI_file = chords_to_MIDI(chords,  bound_times, n_oct_up=st.session_state.n_oct_up, fname='test',
                                   foldername='', microtonal=st.session_state.microtonal, max_beat_duration=8,
                                   total_duration=total_duration)
        st.write("Chords extracted successfully!")


        # Convert MIDI file to byte stream for download
        midi_bytes = io.BytesIO()
        MIDI_file.save("temp.mid")  # Save the MIDI file to a temporary path
        # **Store XML and MIDI in session state**
        st.session_state["xml_bytes"] = xml_bytes
        st.session_state["midi_file"] = MIDI_file

        # Read the saved file into a byte stream
        with open("temp.mid", "rb") as f:
            midi_bytes.write(f.read())

        midi_bytes.seek(0)  # Move to the beginning of the file

    # Always display the stored plot if it exists
    if "chords_plot" in st.session_state:
        st.pyplot(st.session_state["chords_plot"])

    # Show download buttons only if the data exists
    if "xml_bytes" in st.session_state:
        st.download_button(
            label="🎼 Download MusicXML File",
            data=st.session_state["xml_bytes"],
            file_name="chord_progression.musicxml",
            mime="application/vnd.recordare.musicxml"
        )

    if "midi_file" in st.session_state:
        midi_bytes = io.BytesIO()
        st.session_state["midi_file"].save("temp.mid")
        with open("temp.mid", "rb") as f:
            midi_bytes.write(f.read())
        midi_bytes.seek(0)

        st.download_button(
            label="💾 Download MIDI File",
            data=midi_bytes,
            file_name="generated_chords.mid",
            mime="audio/midi"
        )

    if "xml_bytes" in st.session_state and st.button("Show MusicXML Score"):
        # Convert MusicXML to base64
        xml_data = st.session_state["xml_bytes"].getvalue().decode()
        musicxml_b64 = base64.b64encode(xml_data.encode()).decode()

        # Adjust Verovio settings for better rendering
        verovio_html = f"""
        <iframe width="100%" height="700px" 
            style="border: none; overflow: hidden;"
            src="data:text/html;base64,{base64.b64encode(f'''
            <html>
            <head>
                <script src="https://www.verovio.org/javascript/latest/verovio-toolkit.js"></script>
            </head>
            <body>
                <div id="music-container"></div>
                <script>
                    let vrvToolkit = new verovio.toolkit();
                    let musicXmlData = atob("{musicxml_b64}");
                    
                    // Render Verovio with adaptive page height
                    let options = {{
                        pageHeight: 2000,  // Increase for better fitting
                        pageWidth: 1200, 
                        scale: 60,  // Adjust to fit more notation
                        ignoreLayout: 1
                    }};
                    
                    let svgData = vrvToolkit.renderData(musicXmlData, options);
                    document.getElementById("music-container").innerHTML = svgData;
                </script>
            </body>
            </html>'''.encode()).decode()}">
        </iframe>
        """

        st.markdown(verovio_html, unsafe_allow_html=True)

# # --- Rhythms Section ---
# with tab3:
#     st.subheader("Rhythmic Patterns (Coming Soon)")

#     # --- Interactive Wheels ---
#     st.markdown("---")
#     st.subheader("🎛 Harmonic Spectrum Wheel")

    

#     st.subheader("🎚 Frequency Ratio Wheel")

#     wheel_options2 = {
#         "series": [
#             {
#                 "type": "gauge",
#                 "detail": {"formatter": "{value} Hz"},
#                 "data": [{"value": 432, "name": "Base Frequency"}],
#                 "axisLine": {
#                     "lineStyle": {
#                         "width": 12,
#                         "color": [[0.3, "#0099ff"], [0.7, "#9900ff"], [1, "#ff0099"]]
#                     }
#                 },
#             }
#         ]
#     }
#     st_echarts(wheel_options2, height="250px")

with tab3:
    st.subheader("🎨 Color Palette from Tuning")

    if "tuning" in st.session_state:
        fundamental_freq = st.number_input("Fundamental Frequency (Hz)", min_value=100, max_value=1000, value=440)
        
        hsv_colors, hex_colors, color_names = tuning_to_colors(st.session_state.tuning, fundamental=fundamental_freq)

        # Display the colors in a horizontal bar
        fig = go.Figure()

        for i, (color, name) in enumerate(zip(hex_colors, color_names)):
            fig.add_trace(go.Bar(
                y=[1], 
                x=[i], 
                marker=dict(color=color),
                showlegend=False,
                hovertext=f"{name} ({color})"
            ))

        fig.update_layout(
            title="Generated Color Palette",
            xaxis=dict(showticklabels=False),
            yaxis=dict(showticklabels=False),
            height=100,
            margin=dict(l=10, r=10, t=40, b=10)
        )

        st.plotly_chart(fig, use_container_width=True)

        # Display HEX codes and names
        st.write("### Extracted Colors:")
        color_table = [{"Color Name": name, "HEX": hex_color} for name, hex_color in zip(color_names, hex_colors)]
        st.table(color_table)

        # --- Export Section ---
        st.subheader("🎨 Export Color Palette")

        # Choose format
        export_format = st.selectbox("Select export format:", ["ASE", "JSON", "SVG", "CSS", "GPL"])

        # Export button
        if st.button("Export Color Palette"):

            if "tuning" in st.session_state:
                _, hex_colors, color_names = tuning_to_colors(st.session_state.tuning, fundamental=fundamental_freq)

                # Prepare colors dict
                colors = {name: hex_color for name, hex_color in zip(color_names, hex_colors)}

                # Save file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{export_format.lower()}")

                if export_format == "ASE":
                    export_ase(colors, temp_file.name)
                elif export_format == "JSON":
                    export_json(colors, temp_file.name)
                elif export_format == "SVG":
                    export_svg(colors, temp_file.name)
                elif export_format == "CSS":
                    export_css(colors, temp_file.name)
                elif export_format == "GPL":
                    export_gpl(colors, temp_file.name)

                # Provide download button
                with open(temp_file.name, "rb") as file:
                    st.download_button(
                        label=f"💾 Download {export_format} File",
                        data=file,
                        file_name=f"color_palette.{export_format.lower()}",
                        mime="application/octet-stream"
                    )

# Footer
st.markdown("---")
st.markdown("🔬 **Biotuner v0.0.16** | 🎵 Designed for Harmonic Analysis | 🚀 Developed by Antoine Bellemare")



## Deploying the app

# gcloud builds submit --tag gcr.io/kairos-creation-1728503102592/biotuner-gui && \
# gcloud run deploy biotuner-gui \
#   --image gcr.io/kairos-creation-1728503102592/biotuner-gui \
#   --platform managed \
#   --region us-central1 \
#   --memory=2Gi \
#   --cpu=2 \
#   --timeout=100s \
#   --allow-unauthenticated
