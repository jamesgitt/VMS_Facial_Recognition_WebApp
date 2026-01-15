# Complete Deployment Guide

This guide covers everything you need to know to start and deploy the VMS Facial Recognition System.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start (Docker)](#quick-start-docker)
3. [Local Development Setup](#local-development-setup)
4. [Production Deployment](#production-deployment)
5. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

1. **Docker Desktop** (for containerized deployment)
   - Download: https://www.docker.com/products/docker-desktop/
   - Windows: See [database/DOCKER_INSTALLATION.md](./database/DOCKER_INSTALLATION.md)
   - Verify: `docker --version` and `docker compose version`

2. **Node.js 20+** (for local frontend development)
   - Download: https://nodejs.org/
   - Verify: `node --version`

3. **Python 3.11 or 3.12** (for local backend development)
   - Download: https://www.python.org/downloads/
   - Verify: `python --version`
   - ⚠️ Note: Python 3.14 is not supported by TensorFlow

4. **Git** (for cloning the repository)
   - Download: https://git-scm.com/downloads
   - Verify: `git --version`

### Optional Software

- **pgAdmin** (database management) - Included in Docker setup
- **Postman** or **curl** (for API testing)

---

## Quick Start (Docker) - Recommended

This is the fastest way to get the entire system running.

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd VMS_Facial_Recognition_TEST
```

### Step 2: Create Environment File

Create a `.env` file in the project root:

```bash
# Copy the template (or create manually)
# See ENV_TEMPLATE.md for all available options
```

**Minimum required `.env` file:**

```env
# Database Configuration
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_strong_password_here
POSTGRES_DB=visitors_db
POSTGRES_PORT=5432

# Face Recognition API
USE_DATABASE=true
CORS_ORIGINS=http://localhost:3000,http://localhost:3001
API_PORT=8000

# Frontend
FRONTEND_PORT=3000
NEXT_PUBLIC_API_URL=http://localhost:8000
```

**⚠️ Important**: Change `POSTGRES_PASSWORD` to a strong password!

### Step 3: Start All Services

```bash
# Start everything (database + backend + frontend)
docker compose up -d

# View logs
docker compose logs -f

# Check status
docker compose ps
```

### Step 4: Verify Services

1. **Database**: 
   ```bash
   docker exec -it facial_recog_postgres psql -U postgres -d visitors_db -c "SELECT version();"
   ```

2. **Backend API**: 
   - Health check: http://localhost:8000/api/v1/health
   - API docs: http://localhost:8000/docs

3. **Frontend**: 
   - Web app: http://localhost:3000

### Step 5: Access the Application

- **Frontend**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **Database Admin (pgAdmin)**: http://localhost:5050 (if started with `--profile tools`)

### Common Docker Commands

```bash
# Start all services
docker compose up -d

# Stop all services
docker compose down

# View logs
docker compose logs -f              # All services
docker compose logs -f backend     # Specific service
docker compose logs -f postgres    # Database only

# Restart a service
docker compose restart backend

# Rebuild and restart
docker compose up -d --build

# Stop and remove volumes (⚠️ deletes data)
docker compose down -v
```

---

## Local Development Setup

If you want to develop locally without Docker (or run services individually):

### Option A: Database Only (Docker) + Local Services

#### 1. Start Database with Docker

```bash
# Start only the database
docker compose up -d postgres

# Or use the separate database file
docker compose -f docker-compose.db.yml up -d
```

#### 2. Set Up Backend (Python)

```bash
# Navigate to backend directory
cd sevices/face-recognition

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create .env file (or set environment variables)
# See ENV_TEMPLATE.md for required variables

# Start the API
python app/main.py --reload
```

**Backend will run on**: http://localhost:8000

#### 3. Set Up Frontend (Node.js)

```bash
# Navigate to frontend directory
cd apps/facial_recog_web_app

# Install dependencies
npm install

# Create .env.local file
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
echo "DATABASE_URL=postgresql://postgres:your_password@localhost:5432/visitors_db" >> .env.local

# Start development server
npm run dev
```

**Frontend will run on**: http://localhost:3000

### Option B: Full Local Setup (No Docker)

#### 1. Install PostgreSQL Locally

- **Windows**: Download from https://www.postgresql.org/download/windows/
- **Linux**: `sudo apt-get install postgresql`
- **Mac**: `brew install postgresql`

#### 2. Create Database

```bash
# Connect to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE visitors_db;

# Run initialization script
\i database/init.sql
```

#### 3. Set Up Backend

```bash
cd sevices/face-recognition
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# Set environment variables
export USE_DATABASE=true
export DATABASE_URL=postgresql://postgres:password@localhost:5432/visitors_db
# ... (see ENV_TEMPLATE.md for all variables)

# Start API
python app/main.py --reload
```

#### 4. Set Up Frontend

```bash
cd apps/facial_recog_web_app
npm install
npm run dev
```

---

## Production Deployment

### Pre-Deployment Checklist

- [ ] Review [SECURITY_AUDIT.md](./documentation/SECURITY_AUDIT.md)
- [ ] Change all default passwords
- [ ] Set `CORS_ORIGINS` to specific domains (not `*`)
- [ ] Generate `AUTH_SECRET` with `openssl rand -base64 32`
- [ ] Use secrets management service
- [ ] Enable HTTPS/TLS
- [ ] Set up monitoring and logging

### Docker Production Deployment

#### 1. Prepare Environment Variables

Create a production `.env` file with strong passwords:

```env
# Database - Use strong passwords!
POSTGRES_USER=postgres
POSTGRES_PASSWORD=<strong_random_password>
POSTGRES_DB=visitors_db
POSTGRES_PORT=5432

# API Configuration
USE_DATABASE=true
CORS_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
API_PORT=8000

# Frontend
FRONTEND_PORT=3000
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
AUTH_SECRET=<generated_with_openssl_rand_base64_32>
```

#### 2. Build and Deploy

```bash
# Build images
docker compose build

# Start services
docker compose up -d

# Verify health
docker compose ps
docker compose logs
```

#### 3. Set Up Reverse Proxy (HTTPS)

Use **nginx** or **Traefik** to add HTTPS:

**Example nginx configuration:**

```nginx
server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /api {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Cloud Deployment Options

#### Option 1: Docker Compose on VPS

1. **Set up VPS** (DigitalOcean, AWS EC2, etc.)
2. **Install Docker**: Follow Docker installation guide
3. **Clone repository**: `git clone <repo-url>`
4. **Configure `.env`**: Set production values
5. **Deploy**: `docker compose up -d`
6. **Set up reverse proxy**: Use nginx or Traefik

#### Option 2: Container Orchestration (Kubernetes)

1. **Build images**: Push to container registry
2. **Create Kubernetes manifests**: Deploy services, databases, etc.
3. **Configure secrets**: Use Kubernetes secrets for sensitive data
4. **Set up ingress**: Configure HTTPS and routing

#### Option 3: Platform as a Service

- **Railway**: Connect GitHub repo, set environment variables
- **Render**: Deploy Docker Compose stack
- **Fly.io**: Deploy with `flyctl deploy`
- **AWS ECS/Fargate**: Use Docker Compose or ECS task definitions

### Database Setup for Production

#### Option 1: Managed Database (Recommended)

- **AWS RDS**: Managed PostgreSQL
- **Google Cloud SQL**: Managed PostgreSQL
- **Azure Database**: Managed PostgreSQL
- **DigitalOcean Managed Databases**: Managed PostgreSQL

**Benefits**: Automatic backups, scaling, security updates

#### Option 2: Self-Hosted Database

```bash
# Use Docker Compose with production settings
docker compose up -d postgres

# Or install PostgreSQL directly on server
# Configure backups, replication, etc.
```

### Environment Variables for Production

See [ENV_TEMPLATE.md](./ENV_TEMPLATE.md) for complete list. Key production variables:

```env
# Security
POSTGRES_PASSWORD=<strong_password>
PGADMIN_PASSWORD=<strong_password>
CORS_ORIGINS=https://yourdomain.com
AUTH_SECRET=<openssl_rand_base64_32_output>

# Database (if using external)
DATABASE_URL=postgresql://user:password@host:port/database

# API
USE_DATABASE=true
API_PORT=8000

# Frontend
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
```

---

## Importing Visitor Data

If you have visitor data to import:

### Using the Copy Script

```bash
# Ensure database is running
docker compose up -d postgres

# Set environment variables
export DATABASE_URL=postgresql://postgres:password@localhost:5432/visitors_db

# Run copy script
cd database
python copy_data.py visitors.json

# Or with dry-run to validate first
python copy_data.py visitors.json --dry-run
```

**Note**: The script will auto-detect JSON files on your Desktop if no path is provided.

---

## Troubleshooting

### Docker Issues

#### Port Already in Use

```bash
# Check what's using the port
# Windows:
netstat -ano | findstr :5432
# Linux/Mac:
lsof -i :5432

# Change port in .env file
POSTGRES_PORT=5433
```

#### Container Won't Start

```bash
# Check logs
docker compose logs <service_name>

# Check container status
docker compose ps

# Restart service
docker compose restart <service_name>
```

#### Database Connection Issues

```bash
# Verify database is running
docker compose ps postgres

# Check database logs
docker compose logs postgres

# Test connection
docker exec -it facial_recog_postgres psql -U postgres -d visitors_db
```

### Backend Issues

#### Models Not Found

```bash
# Verify models directory exists
ls sevices/face-recognition/models/

# Check MODELS_PATH environment variable
echo $MODELS_PATH

# Models should be in: sevices/face-recognition/models/
```

#### API Not Responding

```bash
# Check if API is running
curl http://localhost:8000/api/v1/health

# Check logs
docker compose logs backend

# Verify environment variables
docker compose exec backend env | grep DATABASE
```

### Frontend Issues

#### Cannot Connect to API

```bash
# Verify NEXT_PUBLIC_API_URL is set correctly
# Check browser console for errors
# Verify CORS_ORIGINS includes frontend URL
```

#### Build Errors

```bash
# Clear Next.js cache
rm -rf apps/facial_recog_web_app/.next
rm -rf apps/facial_recog_web_app/node_modules

# Reinstall dependencies
cd apps/facial_recog_web_app
npm install
npm run build
```

### Database Issues

#### Connection Refused

1. **Check database is running**: `docker compose ps postgres`
2. **Verify connection string**: Check `DATABASE_URL` in `.env`
3. **Check network**: Services must be on same Docker network
4. **Verify credentials**: Username/password must match

#### Import Errors

```bash
# Run with dry-run first
python database/copy_data.py visitors.json --dry-run

# Check JSON format
# Ensure required fields: id, base64Image, firstName, lastName, etc.
```

---

## Monitoring and Maintenance

### Health Checks

All services have health check endpoints:

- **Backend**: `GET http://localhost:8000/api/v1/health`
- **Database**: `docker exec facial_recog_postgres pg_isready -U postgres`
- **Frontend**: Check if http://localhost:3000 loads

### Logs

```bash
# View all logs
docker compose logs -f

# View specific service
docker compose logs -f backend
docker compose logs -f postgres
docker compose logs -f frontend

# Save logs to file
docker compose logs > logs.txt
```

### Backups

#### Database Backup

```bash
# Create backup
docker exec facial_recog_postgres pg_dump -U postgres visitors_db > backup_$(date +%Y%m%d).sql

# Restore backup
docker exec -i facial_recog_postgres psql -U postgres -d visitors_db < backup_20240101.sql
```

#### Automated Backups

Set up cron job or scheduled task:

```bash
# Daily backup script
#!/bin/bash
BACKUP_DIR="/path/to/backups"
DATE=$(date +%Y%m%d_%H%M%S)
docker exec facial_recog_postgres pg_dump -U postgres visitors_db > "$BACKUP_DIR/backup_$DATE.sql"
# Keep only last 7 days
find "$BACKUP_DIR" -name "backup_*.sql" -mtime +7 -delete
```

### Updates

```bash
# Pull latest code
git pull

# Rebuild and restart
docker compose down
docker compose build
docker compose up -d

# Or update specific service
docker compose build backend
docker compose up -d backend
```

---

## Quick Reference

### Essential Commands

```bash
# Start everything
docker compose up -d

# Stop everything
docker compose down

# View logs
docker compose logs -f

# Restart service
docker compose restart <service_name>

# Rebuild
docker compose up -d --build

# Check status
docker compose ps
```

### Service URLs (Default)

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **Database**: localhost:5432
- **pgAdmin**: http://localhost:5050 (with `--profile tools`)

### Important Files

- **`.env`**: Environment variables (create from `ENV_TEMPLATE.md`)
- **`docker-compose.yml`**: Main Docker configuration
- **`ENV_TEMPLATE.md`**: Environment variable reference
- **`README_DOCKER.md`**: Detailed Docker documentation
- **`documentation/SECURITY_AUDIT.md`**: Security checklist

---

## Next Steps

1. ✅ Set up environment variables (`.env` file)
2. ✅ Start services with `docker compose up -d`
3. ✅ Verify all services are running
4. ✅ Import visitor data (if needed)
5. ✅ Configure for production (see [SECURITY_AUDIT.md](./documentation/SECURITY_AUDIT.md))
6. ✅ Set up HTTPS/reverse proxy
7. ✅ Configure monitoring and backups

---

## Getting Help

- **Documentation**: See `documentation/` folder
- **API Documentation**: http://localhost:8000/docs (when running)
- **Security**: Review [SECURITY_AUDIT.md](./documentation/SECURITY_AUDIT.md)
- **Docker**: See [README_DOCKER.md](./README_DOCKER.md)

---

**Last Updated**: Current  
**For detailed API documentation**: See [documentation/API_DOCUMENTATION.md](./documentation/API_DOCUMENTATION.md)
