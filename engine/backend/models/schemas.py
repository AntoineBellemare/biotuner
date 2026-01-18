"""
Pydantic models for API request/response schemas
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class AnalysisConfig(BaseModel):
    """Configuration for biotuner analysis"""
    session_id: str
    method: str = Field(default="harmonic_recurrence", description="Peak extraction method")
    n_peaks: int = Field(default=5, ge=1, le=50)
    precision: float = Field(default=1.0, ge=0.01, le=10)
    max_freq: Optional[float] = Field(default=100, ge=1)
    tuning_method: str = Field(default="peaks_ratios", description="Tuning computation method")
    max_denominator: int = Field(default=100, ge=1, le=1000)
    n_harm: int = Field(default=10, ge=1, le=256)
    start_time: Optional[float] = None
    end_time: Optional[float] = None


class AnalysisResult(BaseModel):
    """Results from biotuner analysis"""
    peaks: List[float]
    powers: List[float]
    tuning: List[float]
    harmonics: Optional[List[float]] = None
    metrics: Optional[Dict[str, Any]] = None


class ChordConfig(BaseModel):
    """Configuration for chord generation"""
    session_id: str
    n_segments: int = Field(default=64, ge=4, le=512)
    method: str = Field(default="cepstrum")
    n_peaks: int = Field(default=5, ge=3, le=10)
    time_resolution: int = Field(default=10, ge=1, le=100)
    frequency_resolution: int = Field(default=1000, ge=100, le=4096)
    # Advanced peak detection parameters
    prominence: float = Field(default=0.5, ge=0.1, le=2.0, description="Minimum peak prominence")
    # MIDI/Music generation parameters
    n_oct_up: int = Field(default=7, ge=0, le=10, description="Octave shift for MIDI notes")


class ColorPaletteConfig(BaseModel):
    """Configuration for color palette generation"""
    session_id: str
    tuning: Optional[List[float]] = None
    peaks: Optional[List[float]] = None
    powers: Optional[List[float]] = None
    fundamental: float = Field(default=440.0, ge=20, le=20000)
    palette_type: str = Field(default="tuning", description="'tuning' or 'peaks'")


class TuningReductionConfig(BaseModel):
    """Configuration for tuning reduction"""
    session_id: str
    n_steps: int = Field(default=12, ge=1, le=128)
    max_ratio: float = Field(default=2.0, ge=1.0, le=10.0)


class TuningPlaybackRequest(BaseModel):
    """Request for tuning audio playback"""
    session_id: str
    tuning: List[float]
    base_freq: float = Field(default=120.0, ge=20, le=2000)
    duration: float = Field(default=0.5, ge=0.1, le=5.0)


class ChordAudioRequest(BaseModel):
    """Request for chord audio generation"""
    session_id: str
    tuning: List[float]
    num_chords: int = Field(default=3, ge=1, le=20)
    base_freq: float = Field(default=440.0, ge=20, le=20000)
    duration: float = Field(default=1.0, ge=0.1, le=10.0)


class MidiExportRequest(BaseModel):
    """Request for MIDI export"""
    session_id: str
    chords: List[List[float]]
    bound_times: List[float]
    total_duration: Optional[float] = None


class MusicXMLRequest(BaseModel):
    """Request for MusicXML export"""
    session_id: str
    chords: List[List[float]]
    bound_times: List[float]
    total_duration: Optional[float] = None
    n_oct_up: int = Field(default=5, ge=0, le=10)


class PaletteExportRequest(BaseModel):
    """Request for palette export"""
    colors: Dict[str, str]  # {color_name: hex_value}
    filename: str = Field(default="palette")


class SessionState(BaseModel):
    """Session state storage"""
    session_id: str
    filename: str
    data: List[float]
    sampling_rate: float
    duration: float
    analysis_result: Optional[Dict[str, Any]] = None
    csv_data: Optional[Dict[str, List[float]]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    
    class Config:
        arbitrary_types_allowed = True
