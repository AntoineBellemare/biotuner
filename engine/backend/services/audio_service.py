"""
Audio Processing Service
Handles audio file loading and processing
"""

import numpy as np
import io
import librosa
import soundfile as sf
import pandas as pd
from typing import Tuple, Optional


class AudioService:
    """Service for audio file processing"""
    
    def load_audio(self, file_content: io.BytesIO, file_type: str) -> Tuple[np.ndarray, float]:
        """
        Load audio file (WAV or MP3)
        
        Parameters
        ----------
        file_content : BytesIO
            File content in memory
        file_type : str
            File extension (wav or mp3)
            
        Returns
        -------
        tuple : (data, sampling_rate)
        """
        try:
            # Load audio using librosa
            data, sr = librosa.load(file_content, sr=None, mono=True)
            return data, float(sr)
        
        except Exception as e:
            print(f"Error loading audio: {str(e)}")
            raise
    
    def load_csv(self, file_content: io.BytesIO) -> Tuple[np.ndarray, float]:
        """
        Load CSV data file
        
        Expected format:
        - Single column: raw data (assumes 256 Hz default)
        - Two columns: time, data (infers sampling rate)
        - Multiple columns: uses first non-time column
        
        Parameters
        ----------
        file_content : BytesIO
            File content in memory
            
        Returns
        -------
        tuple : (data, sampling_rate)
        """
        try:
            # Read CSV
            df = pd.read_csv(file_content)
            
            # Determine data column
            if len(df.columns) == 1:
                # Single column: raw data
                data = df.iloc[:, 0].values
                sr = 256.0  # Default sampling rate
            
            elif len(df.columns) >= 2:
                # Assume first column is time if it looks like it
                first_col = df.columns[0].lower()
                if 'time' in first_col or 't' == first_col:
                    # Calculate sampling rate from time column
                    time_vals = df.iloc[:, 0].values
                    if len(time_vals) > 1:
                        dt = np.mean(np.diff(time_vals))
                        sr = 1.0 / dt if dt > 0 else 256.0
                    else:
                        sr = 256.0
                    
                    # Use second column as data
                    data = df.iloc[:, 1].values
                else:
                    # No time column, use first column as data
                    data = df.iloc[:, 0].values
                    sr = 256.0
            
            else:
                raise ValueError("CSV file is empty")
            
            # Remove NaNs
            data = data[~np.isnan(data)]
            
            return data.astype(np.float64), float(sr)
        
        except Exception as e:
            print(f"Error loading CSV: {str(e)}")
            raise
    
    def segment_audio(
        self,
        data: np.ndarray,
        sr: float,
        start_time: float = 0,
        end_time: Optional[float] = None
    ) -> np.ndarray:
        """
        Extract time segment from audio
        
        Parameters
        ----------
        data : np.ndarray
            Full audio data
        sr : float
            Sampling rate
        start_time : float
            Start time in seconds
        end_time : float, optional
            End time in seconds
            
        Returns
        -------
        np.ndarray : Segmented data
        """
        start_idx = int(start_time * sr)
        end_idx = int(end_time * sr) if end_time else len(data)
        
        return data[start_idx:end_idx]
    
    def normalize_audio(self, data: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range"""
        max_val = np.abs(data).max()
        if max_val > 0:
            return data / max_val
        return data
    
    def create_tuning_audio(
        self,
        tuning: list,
        base_freq: float = 120.0,
        duration: float = 0.5,
        sample_rate: int = 44100
    ) -> bytes:
        """
        Generate audio playback of tuning ratios (arpeggio style)
        
        Parameters
        ----------
        tuning : list
            List of frequency ratios
        base_freq : float
            Base frequency in Hz
        duration : float
            Duration of each note in seconds
        sample_rate : int
            Audio sample rate
            
        Returns
        -------
        bytes : WAV file data
        """
        audio_segments = []
        
        for ratio in tuning:
            freq = base_freq * ratio
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            
            # Create a richer tone using FM synthesis
            modulator = 0.5 * np.sin(2 * np.pi * 2 * freq * t)
            wave = 0.5 * np.sin(2 * np.pi * (freq + 5 * modulator) * t)
            
            # Apply fade in/out to prevent clicks
            fade_samples = int(sample_rate * 0.05)  # 50ms fade
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            wave[:fade_samples] *= fade_in
            wave[-fade_samples:] *= fade_out
            
            audio_segments.append(wave)
        
        # Concatenate all notes
        full_audio = np.concatenate(audio_segments)
        
        # Convert to WAV format
        buffer = io.BytesIO()
        sf.write(buffer, full_audio, sample_rate, format='WAV')
        buffer.seek(0)
        
        return buffer.read()
