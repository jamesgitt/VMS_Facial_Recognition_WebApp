# Quick Start - Development Mode

## 🚀 Fastest Way to Run

### Windows

```powershell
# Option 1: Use the dev script
.\dev.bat

# Option 2: Manual
cd sevices\face-recognition
python app\main.py --reload
```

### Linux/Mac

```bash
# Option 1: Use the dev script
chmod +x dev.sh
./dev.sh

# Option 2: Manual
cd sevices/face-recognition
python app/main.py --reload
```

---

## 📋 Prerequisites Checklist

- [ ] Python 3.11 or 3.12 installed
- [ ] Dependencies installed: `pip install -r requirements.txt`
- [ ] Models downloaded: `python app/download_models.py`
- [ ] Virtual environment (optional but recommended)

---

## 🎯 One-Line Commands

### First Time Setup

```bash
# Install dependencies
pip install fastapi uvicorn[standard] python-multipart pydantic pillow opencv-python numpy

# Download models
python app/download_models.py

# Run server
python app/main.py --reload
```

### Subsequent Runs

```bash
# Just run it
python app/main.py --reload
```

---

## ✅ Verify It's Working

1. **Server starts** - See startup message in console
2. **Health check** - Visit http://localhost:8000/api/v1/health
3. **API docs** - Visit http://localhost:8000/docs

---

## 🔧 Common Issues

**Models not found?**
```bash
python app/download_models.py
```

**Port in use?**
```bash
python app/main.py --port 8001
```

**Import errors?**
```bash
pip install -r requirements.txt
```

---

For detailed instructions, see [DEV_GUIDE.md](DEV_GUIDE.md)
