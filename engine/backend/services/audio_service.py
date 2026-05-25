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
    
    # ------------------------------------------------------------------
    # CSV sampling-rate detection
    # ------------------------------------------------------------------
    # Common column-name conventions for explicit sample-rate metadata
    # in CSV files. Matched case-insensitively against a stripped version
    # of each column header. When a hit returns a single constant value
    # repeated down the column, that's treated as the authoritative sf.
    _SF_NAMES = (
        "sf", "fs", "sampling_rate", "sample_rate", "samplerate",
        "sample_freq", "sampling_freq", "sample_frequency",
        "sampling_frequency", "rate", "samplerate_hz",
    )
    # Column-name patterns for a time axis. When found AND the column is
    # monotonically increasing, sf is derived from the mean inter-sample
    # delta. The time column is also excluded from the candidate data
    # columns so the user doesn't accidentally pick it as the signal.
    _TIME_NAMES = (
        "time", "t", "timestamp", "time_s", "time_ms", "time_us",
        "secs", "seconds", "milliseconds", "ms",
    )

    @staticmethod
    def _normalise_colname(name: str) -> str:
        """Lowercase + strip + remove spaces / dashes for header matching."""
        return str(name).strip().lower().replace(" ", "_").replace("-", "_")

    @classmethod
    def _detect_sf_from_column(cls, df: "pd.DataFrame") -> Optional[Tuple[float, str]]:
        """Scan ``df`` for a column whose header matches one of the known
        sample-rate names. If a single unique numeric value lives there,
        return ``(sf, source_label)``; otherwise ``None``.

        The "single unique numeric value" rule is intentional — some
        recording tools dump a one-row metadata stripe at the top of
        the CSV with a sample-rate constant. Matching against varying
        per-row values would risk treating a noisy signal column as
        the rate, which would be catastrophic.
        """
        for col in df.columns:
            if cls._normalise_colname(col) in cls._SF_NAMES:
                vals = pd.to_numeric(df[col], errors="coerce").dropna().unique()
                if len(vals) == 1 and float(vals[0]) > 0:
                    return float(vals[0]), f"header '{col}'"
        return None

    @classmethod
    def _detect_sf_from_time_column(
        cls, df: "pd.DataFrame",
    ) -> Optional[Tuple[float, str, str]]:
        """Look for a time-axis column. If found with monotonic increasing
        timestamps, derive sf from the median inter-sample delta and
        return ``(sf, source_label, time_col_name)``.

        Median (not mean) of dt makes the result robust to occasional
        gaps or duplicated timestamps — common in real-world capture data.
        """
        for col in df.columns:
            if cls._normalise_colname(col) in cls._TIME_NAMES:
                vals = pd.to_numeric(df[col], errors="coerce").dropna().values
                if len(vals) < 3:
                    continue
                dts = np.diff(vals)
                # Reject non-monotonic, all-zero, or negative-delta sequences.
                if not np.all(dts > 0):
                    continue
                dt = float(np.median(dts))
                if dt <= 0:
                    continue
                # Heuristic: if the column header mentions 'ms' or 'us',
                # convert to seconds for the rate calc. Otherwise assume
                # seconds (the most common convention).
                norm = cls._normalise_colname(col)
                if "ms" in norm:
                    dt = dt / 1000.0
                elif "us" in norm:
                    dt = dt / 1_000_000.0
                if dt <= 0:
                    continue
                return 1.0 / dt, f"time column '{col}' (median Δt)", str(col)
        return None

    def load_csv(self, file_content: io.BytesIO) -> Tuple[np.ndarray, float, dict]:
        """
        Load CSV data file with smart sample-rate detection.

        Detection precedence (first match wins):
          1. An explicit sample-rate column (``sf``, ``fs``,
             ``sampling_rate``, ``sample_rate``, ``samplerate``, …) holding
             a single constant numeric value.
          2. A time-axis column (``time``, ``t``, ``timestamp``, …) with
             monotonically increasing timestamps — sf is derived from the
             median Δt. Honours ``ms`` / ``us`` suffixes in the header.
          3. Fallback default ``256 Hz``.

        Returns a ``(data, sampling_rate, info)`` triple where ``info``
        carries provenance for the UI:
            ``{
                'sf_source': str,         # 'header sf', 'time column t', 'default'
                'sf_default_used': bool,
                'time_column': Optional[str],  # name to exclude from data picks
                'numeric_columns': List[str],
                'sf_columns': List[str],  # explicit sf-name columns found
             }``
        """
        try:
            df = pd.read_csv(file_content)
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_columns) == 0:
                raise ValueError(
                    "CSV has no numeric columns. Check that your data is "
                    "numeric (not dates or text)."
                )

            time_col_name: Optional[str] = None
            sr: Optional[float] = None
            sf_source: str = "default (no metadata found)"

            # 1) Explicit sf column.
            hit = self._detect_sf_from_column(df)
            if hit is not None:
                sr, sf_source = hit

            # 2) Time column.
            if sr is None:
                t_hit = self._detect_sf_from_time_column(df)
                if t_hit is not None:
                    sr, sf_source, time_col_name = t_hit

            # 3) Fallback.
            if sr is None:
                sr = 256.0

            # Pick the data column: prefer first numeric that is NOT the
            # detected time column AND NOT an sf-name column. Falls back
            # to the first numeric column otherwise.
            sf_cols = [
                c for c in df.columns
                if self._normalise_colname(c) in self._SF_NAMES
            ]
            excluded = set(sf_cols)
            if time_col_name is not None:
                excluded.add(time_col_name)
            data_candidates = [c for c in numeric_columns if c not in excluded]
            if data_candidates:
                data_col = data_candidates[0]
            else:
                data_col = numeric_columns[0]
            data = df[data_col].values
            data = data[~np.isnan(data)]

            info = {
                "sf_source":        sf_source,
                "sf_default_used":  sr == 256.0 and sf_source.startswith("default"),
                "time_column":      time_col_name,
                "numeric_columns":  numeric_columns,
                "sf_columns":       sf_cols,
                "data_column":      data_col,
            }
            return data.astype(np.float64), float(sr), info

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
