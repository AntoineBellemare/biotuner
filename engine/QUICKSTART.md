# 🎵 Biotuner v2 - Quick Start Guide

## What's New?

✅ **Modern React UI** - No more page reloads!  
✅ **FastAPI Backend** - Fast, async, WebSocket support  
✅ **Independent Components** - Each section updates separately  
✅ **Railway Deployment** - Easy cloud hosting  
✅ **Better Performance** - Optimized for responsiveness  

## 🚀 Run Locally in 3 Steps

### 1. Start Backend
```bash
cd engine/backend
pip install -r requirements.txt
python main.py
```
✓ Backend running at http://localhost:8000

### 2. Start Frontend
```bash
cd engine/frontend
npm install
npm run dev
```
✓ Frontend running at http://localhost:5173

### 3. Open Browser
Navigate to http://localhost:5173

## 🎯 Key Features

### Tuning Tab
- Upload WAV/MP3/CSV files
- Select analysis method (Harmonic Recurrence, EMD, etc.)
- View frequency peaks and tuning ratios
- Reduce to specific scale (12-TET, 24-TET, etc.)
- Export as .SCL file

### Chords Tab
- Auto-segment audio into chords
- Visualize chord progression
- Play sample chords
- Export as MIDI

### Biocolors Tab
- Convert tuning to color palette
- View colors mapped to frequencies
- Export in multiple formats (ASE, JSON, SVG, CSS, GPL)

## 🚂 Deploy to Railway

### One-Command Deploy

```bash
# Backend
cd engine/backend
railway login
railway init
railway up

# Frontend
cd ../frontend
railway init
railway up
railway variables set VITE_API_URL=https://your-backend-url.railway.app
```

Your app is now live! 🎉

## 📖 Full Documentation

- [README.md](README.md) - Complete documentation
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment guide
- [Backend API Docs](http://localhost:8000/docs) - Interactive API documentation

## 🆚 Comparison with Streamlit Version

| Feature | Streamlit (v1) | FastAPI+React (v2) |
|---------|----------------|-------------------|
| **Page Reloads** | Full reload on every action | No reloads, reactive |
| **Performance** | Slower | Much faster |
| **UI Flexibility** | Limited | Full control |
| **Real-time Updates** | No | Yes (WebSocket) |
| **Deployment** | Google Cloud | Railway/Cloudflare |
| **Mobile Support** | Basic | Optimized |
| **Customization** | Limited | Unlimited |

## 🛠️ Tech Stack

**Backend:**
- FastAPI (Python web framework)
- WebSocket for real-time updates
- Biotuner library (unchanged)

**Frontend:**
- React 18 (UI framework)
- Vite (build tool)
- Tailwind CSS (styling)
- Recharts (visualizations)

## 💡 Tips

1. **Development**: Use `npm run dev` for hot reload
2. **Production**: Run `npm run build` before deploying
3. **Debugging**: Check browser console and Railway logs
4. **CORS Issues**: Update allowed origins in backend/main.py

## 🐛 Troubleshooting

**Backend won't start?**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Frontend won't start?**
```bash
rm -rf node_modules package-lock.json
npm install
```

**Can't connect to backend?**
- Check backend is running on port 8000
- Verify VITE_API_URL in frontend/.env

## 📞 Support

For issues, check:
1. Browser console (F12)
2. Backend logs
3. [Main README](README.md)
4. [Deployment Guide](DEPLOYMENT.md)

---

**Ready to analyze some harmonics? Let's go! 🎼**
