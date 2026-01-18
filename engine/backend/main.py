"""
FastAPI Backend for Biotuner Engine
Provides REST API and WebSocket endpoints for harmonic analysis
"""

from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import numpy as np
import io
import tempfile
import os
from pathlib import Path

# Import biotuner services
from services.biotuner_service import BiotunerService
from services.audio_service import AudioService
from services.chord_service import ChordService
from services.color_service import ColorService
from models.schemas import (
    AnalysisConfig,
    AnalysisResult,
    ChordConfig,
    ColorPaletteConfig,
    TuningReductionConfig,
    TuningPlaybackRequest,
    SessionState,
    ChordAudioRequest,
    MidiExportRequest,
    MusicXMLRequest,
    PaletteExportRequest
)

# Initialize FastAPI app
app = FastAPI(
    title="Biotuner Engine API",
    description="Harmonic Analysis of Time Series",
    version="2.0.0"
)

# Configure CORS for React frontend
# Get allowed origins from environment or use defaults
allowed_origins = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:5173,http://localhost:3000,https://biotuner-engine.kairos-hive.org,https://biotuner-engine.pages.dev"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
biotuner_service = BiotunerService()
audio_service = AudioService()
chord_service = ChordService()
color_service = ColorService()

# Session storage (in production, use Redis or database)
sessions: Dict[str, SessionState] = {}


# ============================================================================
# Health Check & Info
# ============================================================================

@app.get("/")
async def root():
    """API health check"""
    return {
        "name": "Biotuner Engine API",
        "version": "2.0.0",
        "status": "online"
    }

@app.get("/api/info")
async def get_info():
    """Get API information and available methods"""
    return {
        "peak_methods": ["EMD", "fixed", "harmonic_recurrence", "EIMC", "FOOOF"],
        "supported_formats": ["wav", "mp3", "csv"],
        "max_file_size_mb": 50,
        "default_sampling_rate": 256
    }


@app.get("/api/interval-catalog")
async def get_interval_catalog():
    """Get interval catalog for ratio name matching"""
    try:
        from biotuner.dictionaries import interval_catalog
        
        # Convert to JSON-serializable format
        catalog = []
        for name, ratio in interval_catalog:
            try:
                # Convert sympy Rational to float
                ratio_float = float(ratio)
                catalog.append({
                    "name": name,
                    "ratio": ratio_float
                })
            except:
                pass
        
        return {"intervals": catalog}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# File Upload & Processing
# ============================================================================

@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: Optional[str] = None
):
    """
    Upload and process audio/data file
    Returns session ID and basic file info with preview
    """
    try:
        # Validate file type
        file_ext = file.filename.split('.')[-1].lower()
        if file_ext not in ['wav', 'mp3', 'csv']:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Read file content
        content = await file.read()
        
        # Process based on file type
        columns = None
        if file_ext in ['wav', 'mp3']:
            data, sr = audio_service.load_audio(io.BytesIO(content), file_ext)
        else:  # CSV
            # Read CSV to get columns
            import pandas as pd
            df = pd.read_csv(io.BytesIO(content))
            
            # Filter to only numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_columns) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="No numeric columns found in CSV. Please ensure your file contains numeric data (not dates or text). Detected columns: " + ", ".join(df.columns.tolist())
                )
            
            columns = numeric_columns
            # Use first numeric column as default
            data = df[numeric_columns[0]].values
            sr = 256.0  # Default sampling rate for CSV
        
        # Create or update session
        if not session_id:
            session_id = f"session_{np.random.randint(100000, 999999)}"
        
        # Downsample for preview (max 5000 points)
        preview_data = data[::max(1, len(data) // 5000)].tolist()
        
        # Store full CSV data if needed (only numeric columns)
        csv_data = None
        if file_ext == 'csv':
            import pandas as pd
            df = pd.read_csv(io.BytesIO(content))
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            csv_data = {col: df[col].values.tolist() for col in numeric_columns}
        
        sessions[session_id] = SessionState(
            session_id=session_id,
            filename=file.filename,
            data=data.tolist(),
            sampling_rate=sr,
            duration=len(data) / sr
        )
        
        # Store CSV data separately if exists
        if csv_data:
            sessions[session_id].csv_data = csv_data
        
        return {
            "session_id": session_id,
            "filename": file.filename,
            "duration": len(data) / sr,
            "sampling_rate": sr,
            "data_points": len(data),
            "file_type": file_ext,
            "columns": columns,
            "preview_data": preview_data
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/select-column")
async def select_column(session_id: str, column_index: int):
    """
    Select a different column from CSV data
    """
    try:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[session_id]
        
        # Check if CSV data exists
        if not hasattr(session, 'csv_data') or not session.csv_data:
            raise HTTPException(status_code=400, detail="No CSV data available")
        
        columns = list(session.csv_data.keys())
        if column_index < 0 or column_index >= len(columns):
            raise HTTPException(status_code=400, detail="Invalid column index")
        
        # Update session data with selected column
        column_name = columns[column_index]
        column_data = session.csv_data[column_name]
        
        # Try to convert to numeric array
        try:
            data = np.array(column_data, dtype=float)
        except (ValueError, TypeError) as e:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{column_name}' contains non-numeric data (dates or text). Please select a column with numeric values."
            )
        
        # Remove NaNs
        data = data[~np.isnan(data)]
        
        if len(data) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"Column '{column_name}' has no valid numeric data after removing NaN values."
            )
        
        sessions[session_id].data = data.tolist()
        sessions[session_id].duration = len(data) / session.sampling_rate
        
        # Return preview
        preview_data = data[::max(1, len(data) // 5000)].tolist()
        
        return {
            "preview_data": preview_data,
            "data_points": len(data),
            "duration": len(data) / session.sampling_rate
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Biotuner Analysis
# ============================================================================

@app.post("/api/analyze")
async def analyze_biotuner(config: AnalysisConfig):
    """
    Perform biotuner harmonic analysis
    Returns peaks, tuning, and harmonics
    """
    try:
        # Get session data
        if config.session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[config.session_id]
        data = np.array(session.data)
        
        # Apply time selection if specified
        if config.start_time is not None or config.end_time is not None:
            start_idx = int((config.start_time or 0) * session.sampling_rate)
            end_idx = int((config.end_time or session.duration) * session.sampling_rate)
            data = data[start_idx:end_idx]
        
        # Run biotuner analysis
        result = biotuner_service.analyze(
            data=data,
            sf=session.sampling_rate,
            method=config.method,
            n_peaks=config.n_peaks,
            precision=config.precision,
            max_freq=config.max_freq,
            tuning_method=config.tuning_method,
            max_denominator=config.max_denominator,
            n_harm=config.n_harm
        )
        
        # Store results in session
        session.analysis_result = result
        
        return result
    
    except ValueError as e:
        # User-friendly parameter guidance errors
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Unexpected errors
        error_msg = str(e)
        if 'No peak' in error_msg or 'peaks' in error_msg.lower():
            raise HTTPException(
                status_code=400,
                detail=f"Peak detection failed: {error_msg}. Try adjusting precision, max frequency, or peak extraction method."
            )
        raise HTTPException(status_code=500, detail=f"Analysis error: {error_msg}")


@app.post("/api/tuning-reduction")
async def reduce_tuning(config: TuningReductionConfig):
    """
    Apply tuning reduction to generate a scale
    """
    try:
        if config.session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[config.session_id]
        if not session.analysis_result:
            raise HTTPException(status_code=400, detail="No analysis result found")
        
        # Get original tuning
        original_tuning = session.analysis_result.get('tuning', [])
        
        # Apply reduction
        reduction_result = biotuner_service.reduce_tuning(
            original_tuning,
            n_steps=config.n_steps,
            max_ratio=config.max_ratio
        )
        
        return {
            "original_tuning": original_tuning,
            "reduced_tuning": reduction_result['reduced_tuning'],
            "n_steps": len(reduction_result['reduced_tuning']),
            "original_consonance": reduction_result['original_consonance'],
            "reduced_consonance": reduction_result['reduced_consonance']
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Tuning Playback
# ============================================================================

@app.post("/api/play-tuning")
async def play_tuning_audio(request: TuningPlaybackRequest):
    """
    Generate audio playback of tuning ratios
    """
    try:
        # Generate audio for the tuning
        wav_data = audio_service.create_tuning_audio(
            tuning=request.tuning,
            base_freq=request.base_freq,
            duration=request.duration
        )
        
        return StreamingResponse(
            io.BytesIO(wav_data),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=tuning.wav"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Chord Generation & Playback
# ============================================================================

@app.post("/api/generate-chords")
async def generate_chords(config: ChordConfig):
    """
    Generate chord progression from segments
    """
    try:
        if config.session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[config.session_id]
        data = np.array(session.data)
        
        # Get precision and max_freq from session's analysis result if available
        precision = 0.1
        max_freq = 100.0
        
        if hasattr(session, 'analysis_result') and session.analysis_result:
            # Try to get from stored analysis config
            precision = session.analysis_result.get('precision', 0.1)
            max_freq = session.analysis_result.get('max_freq', 100.0)
        
        # Generate segments and extract chords
        result = chord_service.generate_chords(
            data=data,
            sf=session.sampling_rate,
            n_segments=config.n_segments,
            method=config.method,
            n_peaks=config.n_peaks,
            time_resolution=config.time_resolution,
            frequency_resolution=config.frequency_resolution,
            precision=precision,
            prominence=config.prominence,
            min_freq=1.0,
            max_freq=max_freq,
            n_oct_up=config.n_oct_up
        )
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chord-audio")
async def generate_chord_audio(request: ChordAudioRequest):
    """
    Generate WAV audio file from chord progression
    """
    try:
        # Generate chord audio
        wav_data = chord_service.create_chord_audio(
            tuning=request.tuning,
            num_chords=request.num_chords,
            base_freq=request.base_freq,
            duration=request.duration
        )
        
        return StreamingResponse(
            io.BytesIO(wav_data),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=chords.wav"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/export-midi")
async def export_midi(request: MidiExportRequest):
    """
    Export chord progression as MIDI file
    """
    try:
        midi_data = chord_service.create_midi(
            chords=request.chords,
            bound_times=request.bound_times,
            total_duration=request.total_duration
        )
        
        return StreamingResponse(
            io.BytesIO(midi_data),
            media_type="audio/midi",
            headers={"Content-Disposition": "attachment; filename=biotuner_chords.mid"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/export-musicxml")
async def export_musicxml(request: MusicXMLRequest):
    """
    Export chord progression as MusicXML file
    """
    try:
        xml_data = chord_service.create_musicxml(
            chords=request.chords,
            bound_times=request.bound_times,
            total_duration=request.total_duration,
            n_oct_up=request.n_oct_up
        )
        
        return StreamingResponse(
            io.BytesIO(xml_data),
            media_type="application/vnd.recordare.musicxml+xml",
            headers={"Content-Disposition": "attachment; filename=biotuner_chords.musicxml"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Biocolors
# ============================================================================

@app.post("/api/biocolors")
async def generate_biocolors(config: ColorPaletteConfig):
    """
    Generate color palette from tuning or peaks
    """
    try:
        if config.session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = sessions[config.session_id]
        
        if config.palette_type == "peaks":
            # Generate palette from peaks
            peaks = config.peaks
            powers = config.powers
            
            if not peaks and session.analysis_result:
                peaks = session.analysis_result.get('peaks', [])
                powers = session.analysis_result.get('powers', [])
            
            if not peaks:
                raise HTTPException(status_code=400, detail="No peaks available")
            
            color_palette = color_service.generate_palette_from_peaks(
                peaks=peaks,
                powers=powers
            )
        else:
            # Generate palette from tuning (default)
            tuning = config.tuning
            if not tuning and session.analysis_result:
                tuning = session.analysis_result.get('tuning', [])
            
            if not tuning:
                raise HTTPException(status_code=400, detail="No tuning available")
            
            color_palette = color_service.tuning_to_colors(
                tuning=tuning,
                fundamental=config.fundamental
            )
        
        return color_palette
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/export-palette/{format}")
async def export_palette(
    format: str,
    request: PaletteExportRequest
):
    """
    Export color palette in various formats (ase, json, svg, css, gpl)
    """
    try:
        if format not in ['ase', 'json', 'svg', 'css', 'gpl']:
            raise HTTPException(status_code=400, detail="Unsupported format")
        
        print(f"Exporting palette: format={format}, colors={request.colors}, filename={request.filename}")
        file_data = color_service.export_palette(request.colors, format, request.filename)
        
        mime_types = {
            'ase': 'application/octet-stream',
            'json': 'application/json',
            'svg': 'image/svg+xml',
            'css': 'text/css',
            'gpl': 'text/plain'
        }
        
        print(f"Successfully generated {len(file_data)} bytes for {request.filename}.{format}")
        
        return StreamingResponse(
            io.BytesIO(file_data),
            media_type=mime_types[format],
            headers={"Content-Disposition": f"attachment; filename={request.filename}.{format}"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# WebSocket for Real-time Updates
# ============================================================================

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """
    WebSocket connection for real-time updates during analysis
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_json()
            
            # Handle different message types
            if data.get('type') == 'ping':
                await websocket.send_json({"type": "pong"})
            
            elif data.get('type') == 'progress':
                # Send progress updates during long operations
                if session_id in sessions:
                    await websocket.send_json({
                        "type": "progress",
                        "status": "processing",
                        "message": "Analyzing harmonics..."
                    })
    
    except WebSocketDisconnect:
        print(f"WebSocket disconnected for session {session_id}")


# ============================================================================
# Utility Endpoints
# ============================================================================

@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Get session information"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    return {
        "session_id": session.session_id,
        "filename": session.filename,
        "duration": session.duration,
        "sampling_rate": session.sampling_rate,
        "has_analysis": session.analysis_result is not None
    }


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete session data"""
    if session_id in sessions:
        del sessions[session_id]
        return {"message": "Session deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
