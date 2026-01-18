"""
Chord Generation Service
Handles chord extraction and MIDI/audio generation
"""

import numpy as np
import wave
import tempfile
import io
from typing import List, Tuple, Dict, Any, Optional
import sys
from pathlib import Path
from fractions import Fraction

# Add parent biotuner package to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from biotuner.biotuner_object import compute_biotuner
from biotuner.biotuner_utils import create_midi
import librosa


class ChordService:
    """Service for chord generation and audio synthesis"""
    
    def generate_chords(
        self,
        data: np.ndarray,
        sf: float,
        n_segments: int = 64,
        method: str = "cepstrum",
        n_peaks: int = 5,
        time_resolution: int = 10,
        frequency_resolution: int = 1000,
        precision: float = 0.1,
        prominence: float = 0.5,
        min_freq: float = 0.5,
        max_freq: float = 100.0,
        n_oct_up: int = 7
    ) -> Dict[str, Any]:
        """
        Generate chord progression from audio segments
        
        Parameters
        ----------
        data : np.ndarray
            Audio data
        sf : float
            Sampling rate
        n_segments : int
            Number of segments for agglomerative clustering
        method : str
            Peak extraction method
        n_peaks : int
            Number of peaks per chord
        time_resolution : int
            Frame step in milliseconds
        frequency_resolution : int
            FFT window size
        precision : float
            Peak precision in Hz
        prominence : float
            Minimum peak prominence for detection
        min_freq : float
            Minimum frequency to detect (Hz)
        max_freq : float
            Maximum frequency to detect (Hz)
            
        Returns
        -------
        dict : Chords and boundary times
        """
        try:
            # Compute spectral centroid
            n_fft = frequency_resolution
            hop_length = max(1, int((time_resolution / 1000) * sf))
            
            print(f'Using frequency_resolution={n_fft} samples, time_resolution={time_resolution} ms → hop_length={hop_length} samples')
            
            feature = librosa.feature.spectral_centroid(
                y=data,
                sr=sf,
                n_fft=n_fft,
                hop_length=hop_length
            ).squeeze()
            
            # Check if we can segment
            n_features = feature.shape[0]
            print(f"Feature shape: {feature.shape}, Number of features: {n_features}, Requested segments: {n_segments}")
            
            if n_features < n_segments:
                n_segments = max(2, n_features // 2)
                print(f"Adjusted n_segments to {n_segments}")
            
            # Use librosa's agglomerative segmentation (returns exact n_segments boundaries)
            bounds = librosa.segment.agglomerative(feature, n_segments)
            print(f"Librosa returned {len(bounds)} boundary frames for {n_segments} requested segments")
            
            # Convert from frames to time
            bound_times = librosa.frames_to_time(bounds, sr=sf, hop_length=hop_length)
            
            # Convert to sample indices - filter NaN/Inf before converting
            bound_times_clean = bound_times[~np.isnan(bound_times) & ~np.isinf(bound_times)]
            bound_samples = (bound_times_clean * sf).astype(int)
            print(f"Boundary samples: {len(bound_samples)} points (after filtering NaN/Inf), creating {len(bound_samples)-1} segments")
            
            # Extract segments and track valid boundaries
            # Start at 0 (beginning of audio) and end at len(data) (end of audio)
            segments = []
            valid_boundaries_set = {0}  # Use set to avoid duplicates, start at sample 0
            
            # Add all intermediate boundaries from librosa
            for bound in bound_samples:
                # Only add valid bounds that are strictly between 0 and len(data)
                if 0 < bound < len(data) and not np.isnan(bound) and not np.isinf(bound):
                    valid_boundaries_set.add(int(bound))
            
            # Always add the end of the data
            valid_boundaries_set.add(len(data))
            
            # Convert to sorted array
            valid_boundaries = np.array(sorted(valid_boundaries_set))
            print(f"Valid boundaries: {len(valid_boundaries)} unique points (from 0 to {len(data)}), creating {len(valid_boundaries)-1} segments")
            print(f"First 5 boundaries: {valid_boundaries[:5].tolist()}, Last 5: {valid_boundaries[-5:].tolist()}")
            
            # Extract segments using valid boundaries
            for i in range(len(valid_boundaries) - 1):
                start = valid_boundaries[i]
                end = valid_boundaries[i + 1]
                segment = data[start:end]
                
                # Remove NaNs
                segment = segment[~np.isnan(segment)]
                
                # Keep segments that have at least 2 samples
                if len(segment) >= 2:
                    segments.append(segment)
                else:
                    print(f"Warning: Segment {i} too short ({len(segment)} samples), skipping chord extraction but keeping boundary")
                    # Add empty chord to maintain alignment with boundaries
                    segments.append(segment)  # Keep it for now, will be filtered in chord extraction
            
            print(f"Number of segments after extraction: {len(segments)}, Valid boundaries: {len(valid_boundaries)}")
            
            # Extract chords from segments
            chords = self._extract_chords_from_segments(
                segments=segments,
                sf=sf,
                method=method,
                n_peaks=n_peaks,
                precision=precision,
                prominence=prominence,
                min_freq=min_freq,
                max_freq=max_freq
            )
            
            print(f"Number of chords extracted: {len(chords)}")
            
            # Calculate MIDI note range for preview
            all_freqs = [freq for chord in chords if chord for freq in chord]
            midi_range = {}
            
            if all_freqs:
                shift_factor = 2 ** n_oct_up
                midi_notes = []
                for freq in all_freqs:
                    shifted_freq = freq * shift_factor
                    if shifted_freq > 0:
                        try:
                            midi_note = int(round(69 + 12 * np.log2(shifted_freq / 440.0)))
                            midi_notes.append(midi_note)
                        except (ValueError, OverflowError):
                            continue
                
                if midi_notes:
                    midi_range = {
                        'min': int(np.min(midi_notes)),
                        'max': int(np.max(midi_notes)),
                        'median': int(np.median(midi_notes))
                    }
            
            # Convert valid boundaries to time (in seconds)
            bound_times_sec = valid_boundaries / sf
            
            # Final validation - ensure no NaN or Inf values
            if np.any(np.isnan(bound_times_sec)) or np.any(np.isinf(bound_times_sec)):
                print("ERROR: Found NaN or Inf in bound_times_sec!")
                print(f"NaN indices: {np.where(np.isnan(bound_times_sec))[0].tolist()}")
                print(f"Inf indices: {np.where(np.isinf(bound_times_sec))[0].tolist()}")
                # Filter them out
                valid_mask = ~(np.isnan(bound_times_sec) | np.isinf(bound_times_sec))
                bound_times_sec = bound_times_sec[valid_mask]
                print(f"After filtering: {len(bound_times_sec)} boundary times remain")
            
            print(f"Final: {len(chords)} chords, {len(valid_boundaries)} boundaries, {len(bound_times_sec)} bound_times")
            print(f"Bound times (first 10): {bound_times_sec[:10].tolist()}")
            print(f"Bound times (last 5): {bound_times_sec[-5:].tolist()}")
            print(f"Expected {len(chords)} chords and {len(chords) + 1} boundaries (got {len(valid_boundaries)})")
            print(f"All bound_times valid: {not np.any(np.isnan(bound_times_sec))}")
            
            # Calculate segment durations (should have len(valid_boundaries) - 1 durations)
            segment_durations = [
                (valid_boundaries[i+1] - valid_boundaries[i]) / sf
                for i in range(len(valid_boundaries) - 1)
            ]
            
            # Return results
            return {
                'chords': chords,
                'bound_times': bound_times_sec.tolist(),
                'n_segments': len(chords),
                'method': method,
                'n_oct_up': n_oct_up,
                'midi_range': midi_range,
                'segment_durations': segment_durations
            }
        
        except Exception as e:
            print(f"Error generating chords: {str(e)}")
            raise
    
    def _extract_chords_from_segments(
        self,
        segments: List[np.ndarray],
        sf: float,
        method: str,
        n_peaks: int,
        precision: float = 0.1,
        prominence: float = 0.5,
        min_freq: float = 0.5,
        max_freq: float = 100.0
    ) -> List[List[float]]:
        """Extract chords from audio segments using biotuner"""
        chords = []
        
        for segment in segments:
            try:
                # Run biotuner on segment
                bt = compute_biotuner(
                    sf=sf,
                    peaks_function=method,
                    precision=precision
                )
                
                # Pass FREQ_BANDS and n_peaks to peaks_extraction
                bt.peaks_extraction(
                    segment,
                    FREQ_BANDS=[(min_freq, max_freq)],
                    n_peaks=n_peaks,
                    prominence=prominence,
                    ratios_extension=False
                )
                
                if hasattr(bt, 'peaks') and len(bt.peaks) > 0:
                    chords.append(bt.peaks.tolist())
                else:
                    chords.append([])
            
            except Exception as e:
                print(f"Error extracting chord: {str(e)}")
                chords.append([])
        
        return chords
    
    def create_chord_audio(
        self,
        tuning: List[float],
        num_chords: int = 3,
        base_freq: float = 440.0,
        duration: float = 1.0,
        sample_rate: int = 44100
    ) -> bytes:
        """
        Generate WAV audio from random chords
        
        Parameters
        ----------
        tuning : list
            Tuning ratios
        num_chords : int
            Number of chords to generate
        base_freq : float
            Base frequency
        duration : float
            Duration per chord
        sample_rate : int
            Audio sample rate
            
        Returns
        -------
        bytes : WAV file data
        """
        all_waves = []
        silence = np.zeros(int(sample_rate * 0.2))  # 200ms silence
        
        for _ in range(num_chords):
            # Random chord
            num_notes = np.random.randint(3, 6)
            selected_ratios = np.random.choice(tuning, size=num_notes, replace=False)
            
            # Generate frequencies
            freqs = [base_freq * float(Fraction(str(r))) for r in selected_ratios]
            
            # Generate waveform
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            chord_wave = sum(0.3 * np.sin(2 * np.pi * f * t) for f in freqs)
            
            # Normalize
            chord_wave /= max(abs(chord_wave))
            chord_wave *= 0.5
            
            # Add fade in/out
            fade_duration = 0.1
            fade_samples = int(sample_rate * fade_duration)
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            chord_wave[:fade_samples] *= fade_in
            chord_wave[-fade_samples:] *= fade_out
            
            all_waves.extend([chord_wave, silence])
        
        # Concatenate
        final_wave = np.concatenate(all_waves)
        
        # Convert to bytes
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            # Convert to 16-bit PCM
            wav_data = (final_wave * 32767).astype(np.int16)
            wf.writeframes(wav_data.tobytes())
        
        return wav_buffer.getvalue()
    
    def create_midi(
        self,
        chords: List[List[float]],
        bound_times: List[float],
        total_duration: Optional[float] = None
    ) -> bytes:
        """
        Create MIDI file from chord progression
        
        Parameters
        ----------
        chords : list
            List of chord frequencies
        bound_times : list
            Boundary times in samples
        total_duration : float, optional
            Total duration in seconds
            
        Returns
        -------
        bytes : MIDI file data
        """
        try:
            # Convert to durations
            if total_duration:
                bound_times_sec = np.array(bound_times) / bound_times[-1] * total_duration
            else:
                bound_times_sec = np.array(bound_times) / 1000
            
            durations = [
                bound_times_sec[i+1] - bound_times_sec[i]
                for i in range(len(chords) - 1)
            ]
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as tmp:
                tmp_path = tmp.name
            
            # Use biotuner's create_midi
            create_midi(
                chords=chords,
                durations=durations,
                subdivision=1,
                microtonal=True,
                filename=tmp_path.replace('.mid', '')
            )
            
            # Read MIDI file
            with open(tmp_path, 'rb') as f:
                midi_data = f.read()
            
            # Cleanup
            import os
            os.unlink(tmp_path)
            
            return midi_data
        
        except Exception as e:
            print(f"Error creating MIDI: {str(e)}")
            raise
    
    def create_musicxml(
        self,
        chords: List[List[float]],
        bound_times: List[float],
        total_duration: Optional[float] = None,
        n_oct_up: int = 5
    ) -> bytes:
        """
        Create MusicXML file from chord progression
        
        Parameters
        ----------
        chords : list
            List of chord frequencies
        bound_times : list
            Boundary times in seconds
        total_duration : float, optional
            Total duration
        n_oct_up : int
            Number of octaves to shift up
            
        Returns
        -------
        bytes : MusicXML file data
        """
        try:
            from music21 import stream, chord as m21chord, note
            
            score = stream.Score()
            part = stream.Part()
            
            # Convert bound times to durations
            durations = [
                bound_times[i+1] - bound_times[i]
                for i in range(min(len(chords), len(bound_times) - 1))
            ]
            
            # Convert seconds to quarter note lengths
            # Use a tempo of 60 BPM (1 beat = 1 second) for direct mapping
            # So 1 second = 1 quarter note
            quarterLengths = durations  # Direct 1:1 mapping
            
            print(f"Durations (seconds): {durations[:5]}... → Quarter lengths: {quarterLengths[:5]}...")
            
            # Calculate optimal shift to bring frequencies into audible MIDI range
            # Target: frequencies around 200-1000 Hz (MIDI 60-96 range)
            all_freqs = [f for chord in chords for f in chord if f > 0]
            if all_freqs:
                avg_freq = np.median(all_freqs)
                # Calculate how many octaves to shift to get median around 440 Hz
                optimal_octaves = max(0, int(np.ceil(np.log2(440 / avg_freq))))
                # Use the maximum of user setting and optimal
                n_oct_up = max(n_oct_up, optimal_octaves)
            
            shift_factor = 2 ** n_oct_up
            
            print(f"Creating MusicXML with {len(chords)} chords, median freq: {np.median(all_freqs) if all_freqs else 0:.2f} Hz, octave shift: {n_oct_up}, shift factor: {shift_factor}")
            
            chords_added = 0
            for i, chord_freqs in enumerate(chords):
                if len(chord_freqs) == 0:
                    print(f"Chord {i}: empty, skipping")
                    continue
                
                # Shift frequencies up by octaves
                shifted_freqs = [f * shift_factor for f in chord_freqs]
                
                print(f"Chord {i}: original freqs {chord_freqs[:3]}... → shifted {shifted_freqs[:3]}...")
                
                # Convert to MIDI pitches, ensuring valid range and no duplicates
                midi_pitches = []
                for f in shifted_freqs:
                    if f <= 0:
                        continue
                    try:
                        pitch = round(69 + 12 * np.log2(f / 440.0))
                        # Clamp to valid MIDI range
                        pitch = max(21, min(108, pitch))  # A0 to C8
                        if pitch not in midi_pitches:  # Avoid duplicates
                            midi_pitches.append(pitch)
                    except (ValueError, ZeroDivisionError):
                        continue
                
                if len(midi_pitches) == 0:
                    print(f"Chord {i}: no valid MIDI pitches after conversion")
                    continue
                
                print(f"Chord {i}: MIDI pitches {midi_pitches}")
                
                # Create chord
                music_chord = m21chord.Chord(midi_pitches)
                
                # Use actual duration (1 second = 1 quarter note at 60 BPM)
                chord_duration = quarterLengths[i] if i < len(quarterLengths) else 1.0
                music_chord.duration.quarterLength = chord_duration
                
                part.append(music_chord)
                chords_added += 1
            
            print(f"Total chords added to score: {chords_added}")
            
            score.append(part)
            
            # Write to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".musicxml") as tmp:
                tmp_path = tmp.name
            
            score.write(fmt="musicxml", fp=tmp_path)
            
            # Read back
            with open(tmp_path, "rb") as f:
                xml_data = f.read()
            
            # Cleanup
            import os
            os.unlink(tmp_path)
            
            return xml_data
        
        except Exception as e:
            print(f"Error creating MusicXML: {str(e)}")
            raise
