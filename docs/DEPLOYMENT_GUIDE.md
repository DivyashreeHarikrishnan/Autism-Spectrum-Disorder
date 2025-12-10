# Deployment Guide

## Prerequisites
- Python 3.8 or higher
- pip package manager
- Modern web browser
- 2GB free disk space

## Installation Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
cd src
python train_model.py
```

Expected output:
- Model files in `models/` directory
- Training metrics displayed
- Visualizations saved

### 3. Start Backend Server

**Windows:**
```cmd
cd deployment\local
start_backend.bat
```

**Linux/Mac:**
```bash
cd deployment/local
chmod +x start_backend.sh
./start_backend.sh
```

Backend will run on: `http://localhost:8000`

### 4. Start Frontend Server

Open a NEW terminal:

**Windows:**
```cmd
cd deployment\local
start_frontend.bat
```

**Linux/Mac:**
```bash
chmod +x start_frontend.sh
./start_frontend.sh
```

Frontend will run on: `http://localhost:3000`

### 5. Verify Deployment

Open browser and test:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

## Troubleshooting

### Port Already in Use

**Windows:**
```cmd
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**Linux/Mac:**
```bash
lsof -ti:8000 | xargs kill -9
```

### Module Not Found
```bash
pip install --upgrade -r requirements.txt
```

### Model Not Found
```bash
cd src
python train_model.py
```

### CORS Errors
- Ensure both servers are running
- Check browser console for details
- Verify API URL in frontend/js/api.js

## Production Deployment

For production, consider:
1. Use production WSGI server (gunicorn)
2. Set up reverse proxy (nginx)
3. Enable HTTPS
4. Configure firewall rules
5. Set up logging and monitoring
6. Use Docker for containerization

## Health Checks

Monitor these endpoints:
- `GET /` - API health
- System logs in `logs/` directory
- Model file existence in `models/`