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

def generate_segments(data, n_segments=64, sf=44100):
    feature = librosa.feature.spectral_centroid(y=data, sr=sf, n_fft=1000)

    print('Feature size:', feature.size)
    print('Sf:', sf)
    print('Data size:', data.size)  

    bounds = librosa.segment.agglomerative(feature, n_segments)

    # Convert from frames to time (avoid multiplying by sf)
    bound_times = librosa.frames_to_time(bounds, sr=sf, n_fft=1000)

    print('Bound Times:', bound_times)

    # Ensure bound_times does not exceed the duration of the signal
    bound_times = np.clip(bound_times, 0, len(data) / sf)

    # Convert time (s) to sample indices
    bound_samples = (bound_times * sf).astype(int)

    # Generate segments
    segments = segment_time_series(data, bound_samples)
    segments = [segment[~np.isnan(segment)] for segment in segments]

    return segments, bound_samples


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
        
    Returns
    -------
    chords : list of lists
        List of extracted chords.
    """
    # remove nans from segments
    chords = []
    for segment in segments[1:]:
        #print('segment size:', segment.size)
        if segment.size < 256:
            continue
        if np.isnan(segment).any():
            segment = segment[~np.isnan(segment)]
        #normalize segment between 0 and 1
        segment = (segment - np.nanmin(segment)) / (np.nanmax(segment) - np.nanmin(segment))
        try:
            bt = compute_biotuner(sf=sf, data=segment, peaks_function=method, precision=precision)
            bt.peaks_extraction(min_freq=0.001, max_freq=max_freq, n_peaks=n_peaks, nIMFs=n_peaks)
        except:
            continue
        #print(bt.peaks)

        chord=[]
        if peaks_idxs is not None:
            for i in peaks_idxs:
                if i < len(bt.peaks):
                    chord.append(bt.peaks[i])
        else:
            chord = bt.peaks[1:]
        chords.append(chord)
    return chords

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
        .stTabs [data-baseweb="tab"] { font-size: 24px !important; padding: 18px !important; }
        .big-title { font-size: 26px !important; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Sidebar Configuration ---

st.sidebar.title("⚙️ Biotuner Settings")

# Peak Extraction Methods
st.sidebar.subheader("Peak Extraction")
peak_extraction_method = st.sidebar.radio(
    "Select a method:", 
    ["EMD", "EEG Bands", "Harmonic Recurrence", "Intermodulation Components"]
)

method_mapping = {
    "EMD": "EMD",
    "EEG Bands": "fixed",
    "Harmonic Recurrence": "harmonic_recurrence",
    "Intermodulation Components": "EIMC"
}
# Precision Setting
precision = st.sidebar.select_slider(
    "Precision",
    options=[0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10],
    value=1
)

# Sampling Frequency Input
st.sidebar.subheader("Sampling Frequency")
sampling_frequency = st.sidebar.number_input(
    "Enter the sampling frequency (Hz):", 
    min_value=1, 
    value=44100, 
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
        <h1 class="custom-title">Biotuner Graphical Interface</h1>
        <h3 class="custom-subtitle"><i>Harmonic Analysis of Time Series</i></h3>
        """,
        unsafe_allow_html=True
    )


#st.markdown("Upload your **audio or data file** to analyze its **harmonic properties**.")

# File Upload
uploaded_file = st.file_uploader("Upload your **audio or data file**", type=["wav", "csv", "txt"])

if uploaded_file:
    st.success("✅ File uploaded successfully!")
    
    # --- Process WAV File ---
    if uploaded_file.name.endswith('.wav'):
        st.write("🔊 Processing audio file:", uploaded_file.name)
        
        # Load audio
        y, sr = librosa.load(uploaded_file, sr=None)
        st.session_state.uploaded_data = y
        st.session_state.sampling_rate = sr
        
        # Plot time series
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        librosa.display.waveshow(y, sr=sr, ax=ax, color='#40E0D0')
        ax.set_title("Waveform", color='white')
        ax.set_xlabel("Time (s)", color='white')
        ax.set_ylabel("Amplitude", color='white')
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        # remove gridlines
        ax.grid(False)
        st.pyplot(fig)

    # --- Process CSV File ---
    # --- Process CSV File ---
    elif uploaded_file.name.endswith('.csv'):
        st.write("📊 Processing CSV file:", uploaded_file.name)
        
        # Load CSV and store in session state
        if "csv_data" not in st.session_state:
            st.session_state.csv_data = pd.read_csv(uploaded_file)
        
        # Let user choose column, and store it persistently
        column_to_use = st.selectbox("Select column for time series:", st.session_state.csv_data.columns)

        if column_to_use:  # Only proceed if a column is selected
            st.session_state.uploaded_data = st.session_state.csv_data[column_to_use].values
            st.session_state.sampling_rate = sampling_frequency  # Assume user-defined
        
        # Keep the plot persistent
        if "uploaded_data" in st.session_state and st.session_state.uploaded_data is not None:
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_alpha(0.0)
            ax.patch.set_alpha(0.0)
            ax.plot(st.session_state.uploaded_data, color='#40E0D0')
            ax.set_title("Time Series Plot", color='white')
            ax.set_xlabel("Index", color='white')
            ax.set_ylabel(column_to_use, color='white')
            ax.tick_params(axis='x', colors='white')
            ax.tick_params(axis='y', colors='white')
            ax.grid(False)
            st.pyplot(fig)


    else:
        st.write("Unsupported file format. Please upload a .wav or .csv file.")

# --- Tuning Analysis ---
st.markdown('<p class="big-title">🎼 Harmonic Analysis</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Tuning", "Chords", "Rhythms"])

with tab1:
    st.subheader("Tuning Analysis")

    tuning_method = st.selectbox(
        "Tuning Parameter", 
        ['Peaks Ratios', "Harmonic Fit", "Dissonance Curve"]
    )
    max_denom = st.number_input("Max Denominator", min_value=1, value=100, step=1)

    if st.button("Run Tuning Analysis"):
        if st.session_state.uploaded_data is None:
            st.error("Please upload and process a file first.")
        else:
            # Initialize Biotuner Object
            biotuning = compute_biotuner(
                sf=st.session_state.sampling_rate, 
                peaks_function=method_mapping[peak_extraction_method],
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

            tuning_consonance = []
            for index1 in range(len(tuning)):
                for index2 in range(len(tuning)):
                    if tuning[index1] != tuning[index2]:
                        entry = tuning[index1] / tuning[index2]
                        tuning_consonance.append(dyad_similarity(entry))
            tuning_consonance = np.mean(tuning_consonance)
            st.session_state.tuning_consonance = tuning_consonance

    # --- Plot Tuning Analysis ---
    if "tuning" in st.session_state:
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
        # --- Save Tuning Button ---
    if "tuning" in st.session_state:
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

    # **Reduce Tuning Section**
    if "tuning" in st.session_state:
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

    n_segments = st.slider("Number of Segments", 1, 128, 16)

    n_oct_up = st.slider("Number of Octaves Up", 0, 10, 5)

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
    total_duration = st.number_input("Total Duration (s)", min_value=1, value=30, step=1)

    if st.button("Generate Chords"):

        segments, bound_times = generate_segments(data_,
                                                  n_segments=n_segments, sf=st.session_state.sampling_rate)
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
        st.pyplot(fig)

        chords = extract_chords(segments, method=method_mapping[peak_extraction_method],
                                n_peaks=n_peaks, peaks_idxs=None, precision=precision,
                                max_freq=max_freq, sf=st.session_state.sampling_rate) 
        print('CHORDS', chords, 'NUmber of chords:', len(chords))
        MIDI_file = chords_to_MIDI(chords,  bound_times, n_oct_up=n_oct_up, fname='test',
                                   foldername='', microtonal=microtonal, max_beat_duration=8,
                                   total_duration=total_duration)
        st.write("Chords extracted successfully!")
        # Convert MIDI file to byte stream for download
        midi_bytes = io.BytesIO()
        MIDI_file.save("temp.mid")  # Save the MIDI file to a temporary path

        # Read the saved file into a byte stream
        with open("temp.mid", "rb") as f:
            midi_bytes.write(f.read())

        midi_bytes.seek(0)  # Move to the beginning of the file

        # Create a download button
        st.download_button(
            label="💾 Download MIDI File",
            data=midi_bytes,
            file_name="generated_chords.mid",
            mime="audio/midi"
        )
# --- Rhythms Section ---
with tab3:
    st.subheader("Rhythmic Patterns (Coming Soon)")

    # --- Interactive Wheels ---
    st.markdown("---")
    st.subheader("🎛 Harmonic Spectrum Wheel")

    

    st.subheader("🎚 Frequency Ratio Wheel")

    wheel_options2 = {
        "series": [
            {
                "type": "gauge",
                "detail": {"formatter": "{value} Hz"},
                "data": [{"value": 432, "name": "Base Frequency"}],
                "axisLine": {
                    "lineStyle": {
                        "width": 12,
                        "color": [[0.3, "#0099ff"], [0.7, "#9900ff"], [1, "#ff0099"]]
                    }
                },
            }
        ]
    }
    st_echarts(wheel_options2, height="250px")

# Footer
st.markdown("---")
st.markdown("🔬 **Biotuner v1.0** | 🎵 Designed for Harmonic Analysis | 🚀 Developed by Antoine Bellemare")
