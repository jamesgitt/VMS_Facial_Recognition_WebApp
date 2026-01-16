# Docker Compose Setup Guide

## Overview

The project uses a **single `docker-compose.yml` file** that includes all services:
- **PostgreSQL Database** - Visitor data storage (perfect for local testing!)
- **pgAdmin** (optional) - Database management UI
- **Face Recognition API** - FastAPI backend service
- **Next.js Frontend** - Web application

**💡 Note**: The local Docker database is perfect for testing and development. You only need to deploy a database when deploying services to production. See [LOCAL_VS_DEPLOYED_DB.md](./LOCAL_VS_DEPLOYED_DB.md) for details.

## Quick Start

### Start All Services

```bash
# Start everything (database + backend + frontend)
docker compose up -d

# View logs
docker compose logs -f

# Stop everything
docker compose down
```

### Start Only Database

If you only need the database (for local development):

```bash
# Option 1: Use the separate database file
docker compose -f docker-compose.db.yml up -d

# Option 2: Start only postgres from main file
docker compose up -d postgres
```

### Start with pgAdmin

```bash
# Start all services including pgAdmin
docker compose --profile tools up -d
```

## Service Details

### Services Included

| Service | Container Name | Port | Description |
|---------|---------------|------|-------------|
| `postgres` | `facial_recog_postgres` | 5432 | PostgreSQL database |
| `pgadmin` | `facial_recog_pgadmin` | 5050 | Database admin UI (optional) |
| `backend` | `facial_recog_backend` | 8000 | Face Recognition API |
| `frontend` | `facial_recog_frontend` | 3000 | Next.js web app |

### Service Dependencies

```
postgres (database)
  └── backend (depends on postgres)
      └── frontend (depends on backend)
```

## Environment Variables

Create a `.env` file in the project root (see `ENV_TEMPLATE.md`):

```env
# Database
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_strong_password
POSTGRES_DB=visitors_db
POSTGRES_PORT=5432

# pgAdmin (optional)
PGADMIN_EMAIL=admin@admin.com
PGADMIN_PASSWORD=your_strong_password
PGADMIN_PORT=5050

# API
USE_DATABASE=true
CORS_ORIGINS=http://localhost:3000,http://localhost:3001
API_PORT=8000

# Frontend
FRONTEND_PORT=3000
NEXT_PUBLIC_API_URL=http://localhost:8000

# Authentication (Required for email/password auth)
# Generate with: openssl rand -base64 32
AUTH_SECRET=your-generated-secret-here
```

## Common Commands

### Start Services

```bash
# Start all services
docker compose up -d

# Start specific service
docker compose up -d postgres
docker compose up -d backend
docker compose up -d frontend

# Start with pgAdmin
docker compose --profile tools up -d
```

### Stop Services

```bash
# Stop all services
docker compose down

# Stop and remove volumes (⚠️ deletes data)
docker compose down -v

# Stop specific service
docker compose stop backend
```

### View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f backend
docker compose logs -f postgres
docker compose logs -f frontend
```

### Rebuild Services

```bash
# Rebuild all
docker compose build

# Rebuild specific service
docker compose build backend
docker compose build frontend

# Rebuild and restart
docker compose up -d --build
```

### Check Status

```bash
# List running containers
docker compose ps

# Check service health
docker compose ps --format json | jq '.[] | {name: .Name, status: .State}'
```

## Accessing Services

### Database

```bash
# Connection string
postgresql://postgres:${POSTGRES_PASSWORD}@localhost:5432/visitors_db

# Connect via psql
docker exec -it facial_recog_postgres psql -U postgres -d visitors_db
```

### pgAdmin (if enabled)

- URL: http://localhost:5050
- Email: `${PGADMIN_EMAIL}` (default: `admin@admin.com`)
- Password: `${PGADMIN_PASSWORD}`

### Face Recognition API

- URL: http://localhost:8000
- Health Check: http://localhost:8000/api/v1/health
- API Docs: http://localhost:8000/docs (FastAPI Swagger UI)

### Frontend

- URL: http://localhost:3000

## Development Workflow

### Option 1: Full Stack (Recommended)

```bash
# Start everything
docker compose up -d

# Work on code locally, services auto-reload
# Database: localhost:5432
# API: http://localhost:8000
# Frontend: http://localhost:3000
```

### Option 2: Database Only

```bash
# Start only database
docker compose up -d postgres

# Run API locally (outside Docker)
cd services/face-recognition
python app/main.py

# Run frontend locally (outside Docker)
cd apps/facial_recog_web_app
npm run dev
```

## Authentication Setup

The frontend includes email/password authentication. After starting the services:

1. **Access the frontend**: http://localhost:3000
2. **Register a new account**: Click "Sign in" → "Create account" or go to http://localhost:3000/register
3. **Sign in**: Use your registered email and password

**Required Environment Variable:**
- `AUTH_SECRET` - Generate with: `openssl rand -base64 32`

See [EMAIL_PASSWORD_AUTH_SETUP.md](./EMAIL_PASSWORD_AUTH_SETUP.md) for detailed authentication setup.

---

## Authentication Setup

The frontend includes **email/password authentication** with sign-in and registration pages.

### Quick Setup

1. **Set AUTH_SECRET in `.env`:**
   ```bash
   # Generate secret
   openssl rand -base64 32
   
   # Add to .env
   AUTH_SECRET=your-generated-secret-here
   ```

2. **Start services:**
   ```bash
   docker compose up -d
   ```

3. **Access the app:**
   - Frontend: http://localhost:3000
   - Register: http://localhost:3000/register
   - Sign in: http://localhost:3000/signin

4. **Create your first account:**
   - Go to `/register`
   - Enter email, password (min 6 chars), and optional name
   - You'll be redirected to sign in

See [EMAIL_PASSWORD_AUTH_SETUP.md](./EMAIL_PASSWORD_AUTH_SETUP.md) for detailed authentication documentation.

---

## Troubleshooting

### Port Conflicts

If ports are already in use, change them in `.env`:

```env
POSTGRES_PORT=5433
API_PORT=8001
FRONTEND_PORT=3001
```

### Database Connection Issues

1. **Check database is running:**
   ```bash
   docker compose ps postgres
   ```

2. **Check database logs:**
   ```bash
   docker compose logs postgres
   ```

3. **Verify connection string:**
   ```bash
   docker exec -it facial_recog_postgres psql -U postgres -c "SELECT version();"
   ```

### Frontend Can't Connect to Backend API

1. **Check NEXT_PUBLIC_API_URL:**
   - Should be `http://localhost:8000` (for browser access)
   - NOT `http://backend:8000` (that only works inside Docker network)
   - Browser makes requests from your machine, not from inside Docker

2. **Verify backend is accessible:**
   ```bash
   # Test from your machine
   curl http://localhost:8000/api/v1/health
   ```

3. **Check CORS settings:**
   - Backend `CORS_ORIGINS` should include `http://localhost:3000`
   - Check backend logs: `docker compose logs backend`

### Backend Can't Connect to Database

1. **Check backend logs:**
   ```bash
   docker compose logs backend
   ```

2. **Verify DATABASE_URL:**
   - Should use `postgres` as hostname (Docker service name)
   - Format: `postgresql://user:password@postgres:5432/database`

3. **Check network:**
   ```bash
   docker network inspect facial_recog_network
   ```

### Rebuild After Code Changes

```bash
# Rebuild and restart
docker compose up -d --build backend

# Or rebuild everything
docker compose up -d --build
```

## Production Considerations

### Security

- ⚠️ **Change all default passwords** in `.env`
- ⚠️ **Set CORS_ORIGINS** to specific domains (not `*`)
- ⚠️ **Use secrets management** (AWS Secrets Manager, HashiCorp Vault, etc.)
- ⚠️ **Enable HTTPS/TLS** via reverse proxy (nginx, traefik)

### Performance

- Use connection pooling (already configured)
- Set appropriate `DB_VISITOR_LIMIT` for large databases
- Consider read replicas for high-traffic scenarios
- Use Docker resource limits

### Scaling

```bash
# Scale backend service
docker compose up -d --scale backend=3

# Use load balancer in front
```

## File Structure

```
.
├── docker-compose.yml          # Main compose file (all services)
├── docker-compose.db.yml        # Database-only (optional, for dev)
├── .env                         # Environment variables (not in git)
├── ENV_TEMPLATE.md             # Environment variable template
└── database/
    └── init.sql                # Database schema initialization
```

## Benefits of Single Compose File

✅ **Simpler**: One command to start everything  
✅ **Consistent**: All services use same network and configuration  
✅ **Dependencies**: Automatic service ordering (backend waits for database)  
✅ **Environment**: Shared environment variables  
✅ **Development**: Easy to spin up full stack for testing  

## Optional: Database-Only File

The `docker-compose.db.yml` file is kept for:
- Database-only development scenarios
- Testing database migrations
- Sharing database across multiple projects
- Quick database setup without full stack

You can still use it:
```bash
docker compose -f docker-compose.db.yml up -d
```
