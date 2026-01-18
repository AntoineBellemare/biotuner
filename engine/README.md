# Biotuner v2 - FastAPI + React

Modern reactive web application for harmonic analysis of time series using Biotuner.

## 🏗️ Architecture

### Backend (FastAPI)
- **Location**: `backend/`
- **Framework**: FastAPI with WebSocket support
- **Features**:
  - RESTful API for biotuner analysis
  - Real-time updates via WebSocket
  - File upload and processing (WAV, MP3, CSV)
  - Chord generation and MIDI export
  - Color palette generation from tuning

### Frontend (React + Vite)
- **Location**: `frontend/`
- **Framework**: React 18 with Vite
- **Features**:
  - Modern, responsive UI with Tailwind CSS
  - Independent component updates (no full page reloads)
  - Real-time visualization with Recharts
  - Audio playback and export
  - Drag-and-drop file upload

## 🚀 Quick Start

### Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py
```

Backend will run on http://localhost:8000

### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev
```

Frontend will run on http://localhost:5173

## 📦 Deployment

### Railway Deployment

1. **Create Railway Project**
   ```bash
   railway init
   ```

2. **Deploy Backend**
   ```bash
   cd backend
   railway up
   ```

3. **Deploy Frontend**
   ```bash
   cd frontend
   npm run build
   railway up
   ```

4. **Environment Variables**
   - Backend: No special variables needed
   - Frontend: Set `VITE_API_URL` to your backend URL

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build
```

## 🔧 Configuration

### Backend Configuration
- Edit `backend/main.py` for API settings
- Update CORS origins in `main.py` for production
- Modify `services/` for biotuner logic

### Frontend Configuration
- Edit `frontend/vite.config.js` for build settings
- Update API URL in `frontend/src/services/api.js`
- Customize theme in `frontend/tailwind.config.js`

## 📚 API Documentation

Once the backend is running, visit:
- **Interactive API docs**: http://localhost:8000/docs
- **Alternative docs**: http://localhost:8000/redoc

## 🎨 Features

### Tuning Analysis
- Multiple peak extraction methods
- Configurable precision and frequency ranges
- Tuning reduction and scale generation
- SCL file export

### Chord Generation
- Automatic segmentation
- Chord progression extraction
- MIDI export
- Audio playback

### Biocolors
- Frequency-to-color conversion
- Multiple export formats (ASE, JSON, SVG, CSS, GPL)
- Visual palette display
- Consonance-based coloring

## 🛠️ Technology Stack

**Backend**:
- FastAPI
- Uvicorn
- Pydantic
- NumPy, SciPy
- Librosa
- Biotuner library

**Frontend**:
- React 18
- Vite
- Tailwind CSS
- Recharts
- Axios
- React Dropzone
- Lucide React (icons)

## 📝 Development

### Backend Development
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development
```bash
cd frontend
npm run dev
```

### Building for Production
```bash
# Frontend
cd frontend
npm run build

# Backend (no build needed, just ensure dependencies are installed)
cd backend
pip install -r requirements.txt
```

## 🔍 Project Structure

```
engine/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── requirements.txt        # Python dependencies
│   ├── models/
│   │   └── schemas.py         # Pydantic models
│   └── services/
│       ├── biotuner_service.py # Biotuner analysis
│       ├── audio_service.py    # Audio processing
│       ├── chord_service.py    # Chord generation
│       └── color_service.py    # Color palette
├── frontend/
│   ├── package.json           # Node dependencies
│   ├── vite.config.js        # Vite configuration
│   ├── tailwind.config.js    # Tailwind configuration
│   ├── index.html
│   └── src/
│       ├── main.jsx          # Entry point
│       ├── App.jsx           # Main application
│       ├── services/
│       │   └── api.js        # API client
│       └── components/
│           ├── Header.jsx
│           ├── Sidebar.jsx
│           ├── FileUpload.jsx
│           ├── ModalitySelector.jsx
│           ├── TabsContainer.jsx
│           └── tabs/
│               ├── TuningTab.jsx
│               ├── ChordsTab.jsx
│               └── BiocolorsTab.jsx
└── docker-compose.yml
```

## 🐛 Troubleshooting

### CORS Issues
Update allowed origins in `backend/main.py`:
```python
allow_origins=["http://localhost:5173", "https://your-frontend-domain.com"]
```

### WebSocket Connection Failed
Check that the WebSocket URL in `frontend/src/services/api.js` matches your backend URL.

### Import Errors
Ensure the biotuner package is accessible:
```python
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
```

## 📄 License

Same as main Biotuner project (MIT)

## 👨‍💻 Author

Antoine Bellemare - Biotuner v2 Migration
