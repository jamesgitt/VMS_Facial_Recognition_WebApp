# Vercel Deployment Guide

This guide covers deploying the VMS Facial Recognition System using **Vercel for the frontend** and a separate service for the backend and database.

---

## 📋 Overview

**Architecture:**
- **Frontend (Next.js)**: Deploy to Vercel (free tier available)
- **Backend (FastAPI)**: Deploy to Railway, Render, Fly.io, or VPS
- **Database (PostgreSQL)**: Use managed database or deploy separately

**Why this approach?**
- Vercel is optimized for Next.js and offers a generous free tier
- Backend and database need different hosting (Vercel doesn't support long-running Python services)
- This hybrid approach gives you the best of both worlds

---

## 🚀 Deployment Steps

### Step 1: Deploy Backend & Database

Choose one of these options (all have free tiers):

#### Option A: Railway (Recommended - Easiest)

1. **Create Railway Account**: https://railway.app/
2. **Create New Project**
3. **Add PostgreSQL Service**:
   - Click "New" → "Database" → "PostgreSQL"
   - Note the connection details

4. **Add Backend Service**:
   - Click "New" → "GitHub Repo"
   - Select your repository
   - Set root directory to `sevices/face-recognition`
   - Railway will auto-detect Dockerfile

5. **Configure Environment Variables**:
   ```env
   USE_DATABASE=true
   DATABASE_URL=${{Postgres.DATABASE_URL}}
   DB_HOST=${{Postgres.PGHOST}}
   DB_PORT=${{Postgres.PGPORT}}
   DB_NAME=${{Postgres.PGDATABASE}}
   DB_USER=${{Postgres.PGUSER}}
   DB_PASSWORD=${{Postgres.PGPASSWORD}}
   DB_TABLE_NAME=visitors
   DB_VISITOR_ID_COLUMN=id
   DB_IMAGE_COLUMN=base64Image
   DB_VISITOR_LIMIT=0
   CORS_ORIGINS=https://your-app.vercel.app,https://yourdomain.com
   MODELS_PATH=/app/app/models
   API_HOST=0.0.0.0
   API_PORT=$PORT
   ```

6. **Deploy**: Railway will automatically build and deploy
7. **Get Backend URL**: Copy the public URL (e.g., `https://your-backend.railway.app`)

#### Option B: Render

1. **Create Render Account**: https://render.com/

2. **Create PostgreSQL Database**:
   - New → PostgreSQL
   - Choose plan (free tier available)
   - Note connection details

3. **Create Web Service for Backend**:
   - New → Web Service
   - Connect GitHub repo
   - Settings:
     - **Name**: `facial-recognition-backend`
     - **Root Directory**: `sevices/face-recognition`
     - **Environment**: Docker
     - **Dockerfile Path**: `Dockerfile`
     - **Docker Context**: `sevices/face-recognition`

4. **Environment Variables**:
   ```env
   USE_DATABASE=true
   DATABASE_URL=<from-postgres-service>
   DB_HOST=<postgres-host>
   DB_PORT=5432
   DB_NAME=visitors_db
   DB_USER=<postgres-user>
   DB_PASSWORD=<postgres-password>
   DB_TABLE_NAME=visitors
   DB_VISITOR_ID_COLUMN=id
   DB_IMAGE_COLUMN=base64Image
   DB_VISITOR_LIMIT=0
   CORS_ORIGINS=https://your-app.vercel.app
   MODELS_PATH=/app/app/models
   ```

5. **Deploy**: Render will build and deploy
6. **Get Backend URL**: Copy the public URL

#### Option C: Fly.io (Free Tier Available)

1. **Install Fly CLI**:
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. **Login**:
   ```bash
   fly auth login
   ```

3. **Create App**:
   ```bash
   cd sevices/face-recognition
   fly launch
   ```

4. **Add PostgreSQL**:
   ```bash
   fly postgres create --name facial-recognition-db
   fly postgres attach facial-recognition-db
   ```

5. **Set Environment Variables**:
   ```bash
   fly secrets set USE_DATABASE=true
   fly secrets set CORS_ORIGINS=https://your-app.vercel.app
   # DATABASE_URL is automatically set by postgres attach
   ```

6. **Deploy**:
   ```bash
   fly deploy
   ```

7. **Get Backend URL**: `https://your-app.fly.dev`

**Free Tier**: 3 shared VMs, 3GB storage, 160GB outbound data transfer

---

#### Option D: Supabase (Free Database) + Render/Fly.io (Free Backend)

**Best for**: Completely free setup with generous database limits

**Step 1: Set Up Supabase Database (Free)**

1. **Create Supabase Account**: https://supabase.com/
2. **Create New Project**:
   - Click "New Project"
   - Choose organization
   - Set project name
   - Set database password (save it!)
   - Choose region closest to you
   - Click "Create new project"

3. **Get Connection Details**:
   - Go to Project Settings → Database
   - Copy the connection string (URI format)
   - Format: `postgresql://postgres:[YOUR-PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres`

4. **Initialize Schema**:
   - Go to SQL Editor
   - Run the SQL from `database/init.sql`:
   ```sql
   CREATE TABLE IF NOT EXISTS visitors (
       id SERIAL PRIMARY KEY,
       "firstName" VARCHAR(255),
       "lastName" VARCHAR(255),
       "fullName" VARCHAR(255),
       email VARCHAR(255),
       phone VARCHAR(50),
       "imageUrl" TEXT,
       "base64Image" TEXT,
       "createdAt" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
       "updatedAt" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   ```

**Step 2: Deploy Backend to Render (Free Tier)**

1. **Create Render Account**: https://render.com/
2. **Create Web Service**:
   - New → Web Service
   - Connect GitHub repo
   - Settings:
     - **Name**: `facial-recognition-backend`
     - **Root Directory**: `sevices/face-recognition`
     - **Environment**: Docker
     - **Dockerfile Path**: `Dockerfile`
     - **Docker Context**: `sevices/face-recognition`

3. **Environment Variables**:
   ```env
   USE_DATABASE=true
   DATABASE_URL=postgresql://postgres:[PASSWORD]@db.[PROJECT-REF].supabase.co:5432/postgres
   DB_HOST=db.[PROJECT-REF].supabase.co
   DB_PORT=5432
   DB_NAME=postgres
   DB_USER=postgres
   DB_PASSWORD=[YOUR-SUPABASE-PASSWORD]
   DB_TABLE_NAME=visitors
   DB_VISITOR_ID_COLUMN=id
   DB_IMAGE_COLUMN=base64Image
   DB_VISITOR_LIMIT=0
   CORS_ORIGINS=https://your-app.vercel.app
   MODELS_PATH=/app/app/models
   ```

4. **Deploy**: Render will build and deploy automatically

**Free Tier Limits**:
- **Supabase**: 500MB database, 2GB bandwidth, 50,000 monthly active users
- **Render**: Free tier (sleeps after 15min inactivity, wakes on request)

---

#### Option E: Neon (Free PostgreSQL) + Render/Fly.io (Free Backend)

**Best for**: Serverless PostgreSQL with generous free tier

**Step 1: Set Up Neon Database (Free)**

1. **Create Neon Account**: https://neon.tech/
2. **Create Project**:
   - Click "Create Project"
   - Set project name
   - Choose region
   - Click "Create Project"

3. **Get Connection String**:
   - Copy the connection string from dashboard
   - Format: `postgresql://[user]:[password]@[host]/[database]?sslmode=require`

4. **Initialize Schema**:
   - Use Neon SQL Editor or connect via `psql`
   - Run SQL from `database/init.sql`

**Step 2: Deploy Backend** (same as Option D, Step 2)

**Free Tier Limits**:
- **Neon**: 3GB storage, unlimited projects, automatic scaling
- **Render**: Free tier (sleeps after inactivity)

---

#### Option F: Google Cloud Run (Free Tier - 2M Requests/Month)

**Best for**: High-traffic applications with generous free tier

1. **Install Google Cloud SDK**: https://cloud.google.com/sdk/docs/install
2. **Authenticate**:
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

3. **Build and Push Container**:
   ```bash
   cd sevices/face-recognition
   gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/facial-recognition-api
   ```

4. **Deploy to Cloud Run**:
   ```bash
   gcloud run deploy facial-recognition-api \
     --image gcr.io/YOUR_PROJECT_ID/facial-recognition-api \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated \
     --set-env-vars "USE_DATABASE=true,CORS_ORIGINS=https://your-app.vercel.app"
   ```

5. **Set Up Cloud SQL (PostgreSQL)**:
   - Create Cloud SQL instance (free tier available)
   - Connect to Cloud Run service
   - Update environment variables

**Free Tier**: 2 million requests/month, 360,000 GB-seconds, 180,000 vCPU-seconds

---

#### Option G: Replit (Free Tier - Simple Setup)

**Best for**: Quick testing and development

1. **Create Replit Account**: https://replit.com/
2. **Create New Repl**:
   - Import from GitHub
   - Choose Python template
   - Set root to `sevices/face-recognition`

3. **Configure**:
   - Add environment variables in Secrets tab
   - Use Replit Database (free) or external PostgreSQL

4. **Deploy**:
   - Click "Run" button
   - Get public URL

**Note**: Replit free tier has limitations, best for testing only.

---

#### Option H: PythonAnywhere (Free Tier)

**Best for**: Simple Python hosting

1. **Create Account**: https://www.pythonanywhere.com/
2. **Upload Files**:
   - Use Files tab to upload your backend code
3. **Configure Web App**:
   - Create new Web app
   - Choose Flask (or configure for FastAPI)
   - Set up WSGI configuration
4. **Set Environment Variables**:
   - Use `.env` file or environment variables

**Free Tier**: Limited to 1 web app, 512MB disk space, 100,000 requests/day

---

### Step 2: Deploy Frontend to Vercel

#### Option 1: Deploy via Vercel Dashboard (Recommended)

1. **Create Vercel Account**: https://vercel.com/
   - Sign up with GitHub (recommended)

2. **Import Project**:
   - Click "Add New" → "Project"
   - Import your GitHub repository
   - Configure:
     - **Framework Preset**: Next.js (auto-detected)
     - **Root Directory**: `apps/facial_recog_web_app`
     - **Build Command**: `npm run build` (default)
     - **Output Directory**: `.next` (default)

3. **Environment Variables**:
   Click "Environment Variables" and add:
   ```env
   # API URL (from Step 1)
   NEXT_PUBLIC_API_URL=https://your-backend.railway.app
   
   # Database (for Prisma)
   DATABASE_URL=postgresql://user:password@host:5432/visitors_db
   
   # Authentication (generate with: openssl rand -base64 32)
   AUTH_SECRET=your-generated-secret
   
   # Optional: Discord OAuth
   AUTH_DISCORD_ID=your-discord-id
   AUTH_DISCORD_SECRET=your-discord-secret
   ```

4. **Deploy**:
   - Click "Deploy"
   - Vercel will build and deploy automatically
   - Get your frontend URL: `https://your-app.vercel.app`

#### Option 2: Deploy via Vercel CLI

1. **Install Vercel CLI**:
   ```bash
   npm i -g vercel
   ```

2. **Login**:
   ```bash
   vercel login
   ```

3. **Navigate to Frontend Directory**:
   ```bash
   cd apps/facial_recog_web_app
   ```

4. **Deploy**:
   ```bash
   vercel
   ```

5. **Set Environment Variables**:
   ```bash
   vercel env add NEXT_PUBLIC_API_URL
   vercel env add DATABASE_URL
   vercel env add AUTH_SECRET
   ```

6. **Redeploy with Environment Variables**:
   ```bash
   vercel --prod
   ```

---

### Step 3: Configure CORS

Update your backend CORS settings to include your Vercel domain:

```env
# In your backend environment variables
CORS_ORIGINS=https://your-app.vercel.app,https://your-app-git-main.vercel.app,https://your-custom-domain.com
```

**Note**: Vercel creates multiple URLs:
- Production: `https://your-app.vercel.app`
- Preview: `https://your-app-git-branch.vercel.app`
- Custom domain: `https://yourdomain.com`

Add all relevant URLs to CORS_ORIGINS.

---

### Step 4: Set Up Custom Domain (Optional)

1. **In Vercel Dashboard**:
   - Go to your project → Settings → Domains
   - Add your domain (e.g., `yourdomain.com`)
   - Follow DNS configuration instructions

2. **Update Backend CORS**:
   - Add your custom domain to `CORS_ORIGINS`

3. **Update Environment Variables**:
   - Update `NEXT_PUBLIC_API_URL` if using custom domain for API

---

## 🔧 Configuration Files

### Vercel Configuration (`apps/facial_recog_web_app/vercel.json`)

If you need custom configuration, create or update `vercel.json`:

```json
{
  "buildCommand": "npm run build",
  "outputDirectory": ".next",
  "framework": "nextjs",
  "regions": ["iad1"],
  "env": {
    "NEXT_PUBLIC_API_URL": "@next_public_api_url"
  }
}
```

### Next.js Configuration

Ensure `next.config.js` is configured correctly:

```javascript
/** @type {import("next").NextConfig} */
const config = {
  output: 'standalone', // Not needed for Vercel, but good for other platforms
};

export default config;
```

---

## 📝 Environment Variables Summary

### Frontend (Vercel)

| Variable | Description | Example |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | Backend API URL | `https://your-backend.railway.app` |
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:pass@host:5432/db` |
| `AUTH_SECRET` | NextAuth secret | Generated with `openssl rand -base64 32` |
| `AUTH_DISCORD_ID` | Discord OAuth ID (optional) | Your Discord app ID |
| `AUTH_DISCORD_SECRET` | Discord OAuth secret (optional) | Your Discord app secret |

### Backend (Railway/Render/Fly.io)

| Variable | Description | Example |
|----------|-------------|---------|
| `USE_DATABASE` | Enable database | `true` |
| `DATABASE_URL` | PostgreSQL connection string | Auto-set by platform |
| `DB_HOST` | Database host | Auto-set by platform |
| `DB_PORT` | Database port | `5432` |
| `DB_NAME` | Database name | `visitors_db` |
| `DB_USER` | Database user | Auto-set by platform |
| `DB_PASSWORD` | Database password | Auto-set by platform |
| `DB_TABLE_NAME` | Table name | `visitors` |
| `CORS_ORIGINS` | Allowed origins | `https://your-app.vercel.app` |
| `MODELS_PATH` | Path to ML models | `/app/app/models` |

---

## 🗄️ Database Setup

### Initialize Database Schema

After deploying, you need to initialize the database:

1. **Connect to Database**:
   - Use Railway/Render/Fly.io database console
   - Or use `psql` with connection string

2. **Run Initialization Script**:
   ```bash
   # Download init.sql from your repo
   psql $DATABASE_URL < database/init.sql
   ```

   Or manually:
   ```sql
   CREATE TABLE IF NOT EXISTS visitors (
       id SERIAL PRIMARY KEY,
       "firstName" VARCHAR(255),
       "lastName" VARCHAR(255),
       "fullName" VARCHAR(255),
       email VARCHAR(255),
       phone VARCHAR(50),
       "imageUrl" TEXT,
       "base64Image" TEXT,
       "createdAt" TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
       "updatedAt" TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   ```

### Import Visitor Data

If you have existing data:

1. **Use Railway/Render Database Console**:
   - Connect via their web console
   - Import SQL or use `copy_data.py` script

2. **Or Use Local Script**:
   ```bash
   # Set DATABASE_URL to your production database
   export DATABASE_URL=postgresql://user:pass@host:5432/visitors_db
   python database/copy_data.py your-data.json
   ```

---

## ✅ Post-Deployment Verification

1. **Check Frontend**:
   - Visit `https://your-app.vercel.app`
   - Verify page loads
   - Check browser console for errors

2. **Check Backend**:
   - Visit `https://your-backend.railway.app/api/v1/health`
   - Should return: `{"status":"ok","time":"..."}`

3. **Test API Connection**:
   - Open browser console on frontend
   - Check network tab for API calls
   - Verify CORS is working

4. **Test Face Recognition**:
   - Navigate to camera page
   - Allow camera permissions
   - Test face detection

---

## 🔄 Continuous Deployment

### Automatic Deployments

- **Vercel**: Automatically deploys on git push to main branch
- **Railway/Render**: Can be configured for auto-deploy from GitHub

### Manual Deployments

```bash
# Frontend (Vercel)
cd apps/facial_recog_web_app
vercel --prod

# Backend (Railway)
# Push to GitHub, Railway auto-deploys

# Backend (Render)
# Push to GitHub, Render auto-deploys
```

---

## 💰 Cost Estimate (Free Tier Options)

### Completely Free Options

| Service | Free Tier | Notes |
|---------|-----------|-------|
| **Vercel** | ✅ Generous free tier | 100GB bandwidth, unlimited projects |
| **Supabase** | ✅ Free tier | 500MB database, 2GB bandwidth, 50K MAU |
| **Neon** | ✅ Free tier | 3GB storage, unlimited projects |
| **Render** | ✅ Free tier | Sleeps after 15min inactivity, wakes on request |
| **Fly.io** | ✅ Free tier | 3 shared VMs, 3GB storage, 160GB transfer |
| **Google Cloud Run** | ✅ Free tier | 2M requests/month, 360K GB-seconds |
| **Replit** | ✅ Free tier | Limited, best for testing |
| **PythonAnywhere** | ✅ Free tier | 1 web app, 512MB disk, 100K requests/day |

### Recommended Free Combinations

**Option 1: Completely Free (Best for Testing)**
- Frontend: Vercel (free)
- Backend: Render (free, sleeps after inactivity)
- Database: Supabase (free, 500MB)
- **Cost**: $0/month

**Option 2: Always-On Free (Best for Production)**
- Frontend: Vercel (free)
- Backend: Fly.io (free, always-on)
- Database: Neon (free, 3GB)
- **Cost**: $0/month

**Option 3: High-Traffic Free**
- Frontend: Vercel (free)
- Backend: Google Cloud Run (free, 2M requests/month)
- Database: Supabase or Neon (free)
- **Cost**: $0/month (up to 2M requests)

**Option 4: Easiest Setup (Small Cost)**
- Frontend: Vercel (free)
- Backend: Railway ($5 credit/month)
- Database: Railway PostgreSQL (included)
- **Cost**: $0-5/month (usually free with credit)

### Free Tier Limitations

**Render Free Tier**:
- ⚠️ Sleeps after 15 minutes of inactivity
- ⚠️ Takes 30-60 seconds to wake up
- ✅ Good for: Development, low-traffic apps

**Fly.io Free Tier**:
- ✅ Always-on (no sleep)
- ✅ Good for: Production apps
- ⚠️ Limited to 3 shared VMs

**Supabase Free Tier**:
- ✅ 500MB database (enough for ~50K-100K visitors)
- ✅ 2GB bandwidth/month
- ✅ 50,000 monthly active users

**Neon Free Tier**:
- ✅ 3GB database storage
- ✅ Automatic scaling
- ✅ Serverless (pay-per-use after free tier)

---

## 🐛 Troubleshooting

### Frontend Issues

#### Build Fails on Vercel

**Error**: Environment variables not found
- **Solution**: Add all required env vars in Vercel dashboard

**Error**: API calls failing
- **Solution**: Check `NEXT_PUBLIC_API_URL` is correct
- **Solution**: Verify CORS is configured on backend

#### CORS Errors

**Error**: `Access-Control-Allow-Origin` header missing
- **Solution**: Add Vercel URL to backend `CORS_ORIGINS`
- **Solution**: Include all Vercel preview URLs

### Backend Issues

#### Database Connection Fails

**Error**: Connection refused
- **Solution**: Verify `DATABASE_URL` is correct
- **Solution**: Check database is running and accessible
- **Solution**: Verify firewall/security group allows connections

#### Models Not Found

**Error**: `FileNotFoundError: model not found`
- **Solution**: Ensure models are in Docker image
- **Solution**: Check `MODELS_PATH` environment variable
- **Solution**: Verify models directory is copied in Dockerfile

### Deployment Issues

#### Railway/Render Build Fails

**Error**: Docker build fails
- **Solution**: Check Dockerfile is in correct location
- **Solution**: Verify root directory is set correctly
- **Solution**: Check build logs for specific errors

---

## 🔒 Security Considerations

1. **Environment Variables**:
   - Never commit `.env` files
   - Use platform secrets management
   - Rotate secrets regularly

2. **CORS Configuration**:
   - Only allow specific domains (not `*`)
   - Include all Vercel preview URLs if needed
   - Remove test URLs before production

3. **Database Security**:
   - Use strong passwords
   - Enable SSL connections
   - Restrict database access to backend only

4. **API Security**:
   - Consider adding API key authentication
   - Implement rate limiting
   - Monitor for abuse

---

## 📚 Additional Resources

- **Vercel Docs**: https://vercel.com/docs
- **Railway Docs**: https://docs.railway.app/
- **Render Docs**: https://render.com/docs
- **Fly.io Docs**: https://fly.io/docs/

---

## 🎯 Quick Start Checklist

- [ ] Deploy backend to Railway/Render/Fly.io
- [ ] Get backend URL
- [ ] Initialize database schema
- [ ] Deploy frontend to Vercel
- [ ] Set environment variables in Vercel
- [ ] Configure CORS on backend
- [ ] Test frontend → backend connection
- [ ] Test face recognition functionality
- [ ] Set up custom domain (optional)
- [ ] Configure monitoring/alerts

---

## 💡 Tips

1. **Use Railway for Backend**: Easiest setup, good free tier
2. **Use Vercel for Frontend**: Optimized for Next.js, great DX
3. **Test Locally First**: Use `vercel dev` to test locally
4. **Monitor Logs**: Check Vercel and backend logs for issues
5. **Use Preview Deployments**: Test changes before production

---

## Next Steps

After successful deployment:

1. Set up monitoring (Vercel Analytics, Sentry, etc.)
2. Configure custom domain
3. Set up CI/CD pipelines
4. Implement backup strategy for database
5. Add rate limiting and API authentication
6. Set up alerts for service failures
