# Environment Variables Template for docker-compose.yml

This file contains all environment variables needed for the root `docker-compose.yml` file.

**⚠️ NEVER commit `.env` files to version control!**

Copy the variables below to your `.env` file in the project root and fill in your values.

## Required Environment Variables

### Database Configuration

```env
# PostgreSQL Database
POSTGRES_USER=postgres
POSTGRES_PASSWORD=CHANGE_ME_STRONG_PASSWORD
POSTGRES_DB=visitors_db
POSTGRES_PORT=5432

# pgAdmin (optional - only needed if using --profile tools)
PGADMIN_EMAIL=admin@admin.com
PGADMIN_PASSWORD=CHANGE_ME_STRONG_PASSWORD
PGADMIN_PORT=5050
```

### Face Recognition API (Backend)

```env
# Database Integration
USE_DATABASE=true

# Database Table Configuration
DB_TABLE_NAME=visitors
DB_VISITOR_ID_COLUMN=id
DB_IMAGE_COLUMN=base64Image
DB_VISITOR_LIMIT=0

# API Configuration
CORS_ORIGINS=http://localhost:3000,http://localhost:3001,http://frontend:3000
API_PORT=8000
```

**Note**: The `DATABASE_URL` and database connection parameters are automatically constructed in `docker-compose.yml` using the PostgreSQL service name and the variables above. The backend service connects to the database using the Docker service name `postgres` as the hostname.

### Next.js Frontend

```env
# Frontend Configuration
FRONTEND_PORT=3000

# API URL (for browser to access backend)
# Use http://localhost:8000 - browser makes requests from your machine
# NOT http://backend:8000 (that only works inside Docker network)
NEXT_PUBLIC_API_URL=http://localhost:8000

# Authentication (Required for email/password auth)
# Generate with: openssl rand -base64 32
AUTH_SECRET=CHANGE_ME_GENERATE_WITH_openssl_rand_base64_32
```

## Complete .env File Example

```env
# ============================================
# Database Configuration
# ============================================
POSTGRES_USER=postgres
POSTGRES_PASSWORD=CHANGE_ME_STRONG_PASSWORD
POSTGRES_DB=visitors_db
POSTGRES_PORT=5432

PGADMIN_EMAIL=admin@admin.com
PGADMIN_PASSWORD=CHANGE_ME_STRONG_PASSWORD
PGADMIN_PORT=5050

# ============================================
# Face Recognition API (Backend)
# ============================================
USE_DATABASE=true
DB_TABLE_NAME=visitors
DB_VISITOR_ID_COLUMN=id
DB_IMAGE_COLUMN=base64Image
DB_VISITOR_LIMIT=0
CORS_ORIGINS=http://localhost:3000,http://localhost:3001,http://frontend:3000
API_PORT=8000

# ============================================
# Next.js Frontend
# ============================================
FRONTEND_PORT=3000
# API URL for browser access (always use localhost, not Docker service name)
NEXT_PUBLIC_API_URL=http://localhost:8000
# Authentication secret (generate with: openssl rand -base64 32)
AUTH_SECRET=CHANGE_ME_GENERATE_WITH_openssl_rand_base64_32
```

## Usage

1. Copy this template to `.env` in the project root:
   ```bash
   # Create .env file and edit with your values
   # (Copy the complete example above)
   ```

2. Start services:
   ```bash
   docker compose up -d
   ```

3. Services will automatically use these environment variables.

## Production Checklist

Before deploying to production:

- [ ] Change all default passwords
- [ ] Use strong, unique passwords (min 16 characters)
- [ ] Set `CORS_ORIGINS` to specific domains (not `*`)
- [ ] Generate `AUTH_SECRET` with `openssl rand -base64 32`
- [ ] Use secrets management service (AWS Secrets Manager, HashiCorp Vault, etc.)
- [ ] Enable HTTPS/TLS via reverse proxy
- [ ] Review all default values
- [ ] Remove or secure debug endpoints

## Notes

- **Database Connection**: The backend automatically connects to the database using the Docker service name `postgres` as the hostname. No need to set `DB_HOST`, `DB_PORT`, etc. manually - they're configured in `docker-compose.yml`.
- **Service Communication**: Services communicate using Docker service names (e.g., `postgres`, `backend`, `frontend`) on the internal network.
- **Port Mapping**: The `*_PORT` variables control which ports are exposed on your host machine.
