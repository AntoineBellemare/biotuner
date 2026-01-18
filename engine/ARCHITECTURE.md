# Biotuner v2 - Architecture Overview

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          USER BROWSER                            │
│                    (http://localhost:5173)                       │
└───────────────────┬─────────────────────────────────────────────┘
                    │
                    │ HTTP/WebSocket
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    REACT FRONTEND (Vite)                         │
├─────────────────────────────────────────────────────────────────┤
│  Components:                                                     │
│  ├─ App.jsx                    (Main application)               │
│  ├─ Header.jsx                 (Logo & title)                   │
│  ├─ Sidebar.jsx                (Settings panel)                 │
│  ├─ FileUpload.jsx             (Drag & drop)                    │
│  ├─ ModalitySelector.jsx       (Signal type picker)             │
│  └─ TabsContainer.jsx          (Tab system)                     │
│      ├─ TuningTab.jsx          (Harmonic analysis)              │
│      ├─ ChordsTab.jsx          (Chord generation)               │
│      └─ BiocolorsTab.jsx       (Color palettes)                 │
│                                                                  │
│  Services:                                                       │
│  └─ api.js                     (HTTP client + WebSocket)        │
└───────────────────┬───────────────────────────────────────────┘
                    │
                    │ REST API / WebSocket
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FASTAPI BACKEND (Uvicorn)                      │
│                    (http://localhost:8000)                       │
├─────────────────────────────────────────────────────────────────┤
│  Endpoints:                                                      │
│  ├─ POST /api/upload           (File upload)                    │
│  ├─ POST /api/analyze          (Harmonic analysis)              │
│  ├─ POST /api/tuning-reduction (Scale reduction)                │
│  ├─ POST /api/generate-chords  (Chord extraction)               │
│  ├─ POST /api/chord-audio      (Audio generation)               │
│  ├─ POST /api/export-midi      (MIDI export)                    │
│  ├─ POST /api/biocolors        (Color palette)                  │
│  ├─ POST /api/export-palette   (Palette export)                 │
│  ├─ GET  /api/session/:id      (Session info)                   │
│  └─ WS   /ws/:id               (Real-time updates)              │
│                                                                  │
│  Services:                                                       │
│  ├─ biotuner_service.py        (Analysis wrapper)               │
│  ├─ audio_service.py           (File processing)                │
│  ├─ chord_service.py           (Chord generation)               │
│  └─ color_service.py           (Palette generation)             │
│                                                                  │
│  Models:                                                         │
│  └─ schemas.py                 (Pydantic models)                │
└───────────────────┬───────────────────────────────────────────┘
                    │
                    │ Python imports
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    BIOTUNER LIBRARY                              │
│                  (Existing Python Package)                       │
├─────────────────────────────────────────────────────────────────┤
│  ├─ biotuner_object.py         (Core analysis)                  │
│  ├─ biotuner_utils.py          (Utilities)                      │
│  ├─ scale_construction.py      (Tuning reduction)               │
│  ├─ biocolors.py               (Color conversion)               │
│  ├─ metrics.py                 (Consonance, similarity)         │
│  └─ harmonic_spectrum.py       (Spectral analysis)              │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 Data Flow

### 1. File Upload Flow
```
User selects file
    ↓
FileUpload.jsx (drag & drop)
    ↓
api.uploadFile()
    ↓
POST /api/upload
    ↓
AudioService.load_audio() or load_csv()
    ↓
Store in session
    ↓
Return session_id + file_info
    ↓
Update React state
    ↓
Display file info
```

### 2. Analysis Flow
```
User clicks "Analyze Harmonics"
    ↓
handleAnalyze() in App.jsx
    ↓
api.analyze(config)
    ↓
POST /api/analyze
    ↓
BiotunerService.analyze()
    ↓
compute_biotuner() [Biotuner lib]
    ↓
Extract peaks, tuning, harmonics
    ↓
Return analysis result
    ↓
Update React state
    ↓
TuningTab displays results
```

### 3. Chord Generation Flow
```
User configures settings
    ↓
ChordsTab.jsx
    ↓
api.generateChords(config)
    ↓
POST /api/generate-chords
    ↓
ChordService.generate_chords()
    ↓
librosa segmentation
    ↓
compute_biotuner() per segment
    ↓
Return chord progression
    ↓
Display chords
    ↓
User clicks "Play" or "Export MIDI"
```

### 4. Color Palette Flow
```
User sets fundamental frequency
    ↓
BiocolorsTab.jsx
    ↓
api.generateBiocolors(config)
    ↓
POST /api/biocolors
    ↓
ColorService.tuning_to_colors()
    ↓
audible2visible() [Biotuner lib]
    ↓
wavelength_to_rgb()
    ↓
Return color palette
    ↓
Display colors
    ↓
User exports in desired format
```

## 🔄 Real-time Updates (WebSocket)

```
Frontend connects
    ↓
ws = new WebSocket('/ws/:session_id')
    ↓
Backend accepts connection
    ↓
Long-running operation starts
    ↓
Backend sends progress updates
    ↓
Frontend receives & displays progress
    ↓
Operation completes
    ↓
Backend sends final result
    ↓
Frontend updates UI
```

## 🗄️ Session Management

```
┌──────────────────────┐
│   Session Storage    │
│  (In-memory dict)    │
├──────────────────────┤
│  session_123: {      │
│    data: [...]       │
│    sr: 44100         │
│    filename: "..."   │
│    analysis: {...}   │
│  }                   │
└──────────────────────┘
```

**Production**: Replace with Redis or database

## 🌐 Deployment Architecture

### Railway Deployment
```
┌─────────────────────────────────────────────────┐
│                  Railway Platform                │
├─────────────────────────────────────────────────┤
│                                                  │
│  ┌──────────────────┐    ┌──────────────────┐  │
│  │  Backend Service │    │ Frontend Service │  │
│  │                  │    │                  │  │
│  │  FastAPI         │    │  Static Files    │  │
│  │  Python 3.10     │    │  (Vite build)    │  │
│  │  Port: 8000      │    │  Port: 80        │  │
│  └────────┬─────────┘    └────────┬─────────┘  │
│           │                       │             │
│           │  Internal Network     │             │
│           └───────────────────────┘             │
│                                                  │
└─────────────────────────────────────────────────┘
           │                        │
           │ HTTPS                  │ HTTPS
           ▼                        ▼
    api.biotuner.com        biotuner.com
```

### Docker Compose (Local)
```
┌─────────────────────────────────────────────────┐
│              Docker Compose                      │
├─────────────────────────────────────────────────┤
│                                                  │
│  ┌──────────────────┐    ┌──────────────────┐  │
│  │  backend:        │    │  frontend:       │  │
│  │   - Port 8000    │◄───┤   - Port 5173    │  │
│  │   - Hot reload   │    │   - Hot reload   │  │
│  │   - Volume mount │    │   - Proxy API    │  │
│  └──────────────────┘    └──────────────────┘  │
│                                                  │
└─────────────────────────────────────────────────┘
```

## 🎨 Component Hierarchy

```
App.jsx
├── Header.jsx
├── Sidebar.jsx
│   ├── Method Selector
│   ├── Precision Slider
│   ├── Peaks Input
│   └── File Info Display
└── Main Content
    ├── ModalitySelector.jsx
    │   └── 6 Modality Buttons
    ├── FileUpload.jsx
    │   └── Dropzone
    ├── Analyze Button
    └── TabsContainer.jsx
        ├── TuningTab.jsx
        │   ├── Analysis Info Cards
        │   ├── Peaks Chart (Recharts)
        │   ├── Tuning Ratios Grid
        │   ├── Reduction Controls
        │   └── Export Button
        ├── ChordsTab.jsx
        │   ├── Config Form
        │   ├── Chord Stats
        │   ├── Progression List
        │   ├── Audio Player
        │   └── Export Buttons
        └── BiocolorsTab.jsx
            ├── Fundamental Input
            ├── Palette Display
            ├── Color Swatches
            ├── Details Table
            └── Export Buttons
```

## 🔐 Security Architecture

```
Request Flow:
    ↓
1. CORS Validation (FastAPI Middleware)
    ↓
2. File Type Validation (upload endpoint)
    ↓
3. File Size Check (< 50MB)
    ↓
4. Pydantic Schema Validation
    ↓
5. Session Validation
    ↓
6. Process Request
    ↓
7. Return Response
```

## 📈 Scaling Strategy

### Vertical Scaling
```
Railway Service
├── Hobby: 512MB RAM
├── Pro: 8GB RAM
└── Enterprise: Custom
```

### Horizontal Scaling
```
Load Balancer
    ↓
┌───────┬───────┬───────┐
│ App 1 │ App 2 │ App 3 │
└───┬───┴───┬───┴───┬───┘
    └───────┼───────┘
            ↓
      Redis Session Store
```

## 🛠️ Technology Stack Details

### Backend Dependencies
```
fastapi==0.104.1          # Web framework
uvicorn==0.24.0          # ASGI server
pydantic==2.5.0          # Data validation
numpy<2.0                # Arrays
scipy>=1.7.3             # Scientific computing
librosa>=0.9.2           # Audio processing
pandas>=1.3.0            # Data manipulation
scikit-learn>=1.0.0      # ML/clustering
+ biotuner package       # Your library
```

### Frontend Dependencies
```
react@18.2.0             # UI framework
vite@5.0.0              # Build tool
axios@1.6.0             # HTTP client
recharts@2.10.0         # Charts
react-dropzone@14.2.3   # File upload
tailwindcss@3.3.5       # CSS framework
lucide-react@0.294.0    # Icons
```

## 🔍 Monitoring Points

### Health Checks
- Backend: `GET /`
- Frontend: `GET /` (static file)
- WebSocket: Connection test

### Metrics to Track
- Request latency
- Error rates
- Active sessions
- File upload sizes
- Analysis duration
- Memory usage
- CPU usage

### Logging
- Request logs (FastAPI)
- Error logs (Python logging)
- Build logs (Railway)
- Browser console (Frontend)

---

This architecture is designed for:
✅ **Performance** - Fast, async, non-blocking
✅ **Scalability** - Easy to scale horizontally
✅ **Maintainability** - Clean separation of concerns
✅ **Extensibility** - Easy to add features
✅ **Reliability** - Error handling throughout
