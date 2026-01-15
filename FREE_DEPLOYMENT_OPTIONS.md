# Free Deployment Options Guide

Quick reference for completely free deployment options for the VMS Facial Recognition System.

---

## 🎯 Best Free Combinations

### Option 1: Completely Free (Recommended for Testing)

**Stack:**
- **Frontend**: Vercel (free)
- **Backend**: Render (free)
- **Database**: Supabase (free)

**Pros:**
- ✅ $0/month
- ✅ Easy setup
- ✅ Generous free tiers

**Cons:**
- ⚠️ Render sleeps after 15min (takes 30-60s to wake)
- ⚠️ Supabase: 500MB database limit

**Best for**: Testing, development, low-traffic apps

---

### Option 2: Always-On Free (Recommended for Production)

**Stack:**
- **Frontend**: Vercel (free)
- **Backend**: Fly.io (free)
- **Database**: Neon (free)

**Pros:**
- ✅ $0/month
- ✅ Always-on (no sleep)
- ✅ 3GB database (Neon)
- ✅ Good performance

**Cons:**
- ⚠️ Fly.io: Limited to 3 shared VMs
- ⚠️ Setup slightly more complex

**Best for**: Production apps, always-available services

---

### Option 3: High-Traffic Free

**Stack:**
- **Frontend**: Vercel (free)
- **Backend**: Google Cloud Run (free)
- **Database**: Supabase or Neon (free)

**Pros:**
- ✅ $0/month
- ✅ 2 million requests/month free
- ✅ Auto-scaling
- ✅ Always-on

**Cons:**
- ⚠️ Requires Google Cloud account setup
- ⚠️ More complex configuration

**Best for**: High-traffic applications

---

## 📊 Free Tier Comparison

### Frontend Hosting

| Service | Free Tier | Best For |
|---------|-----------|----------|
| **Vercel** | ✅ Unlimited projects, 100GB bandwidth | Next.js apps (recommended) |
| **Netlify** | ✅ 100GB bandwidth, 300 build minutes | Alternative to Vercel |

### Backend Hosting

| Service | Free Tier | Always-On | Notes |
|---------|-----------|-----------|-------|
| **Render** | ✅ Yes | ❌ Sleeps after 15min | Easy setup, wakes on request |
| **Fly.io** | ✅ Yes | ✅ Yes | 3 shared VMs, 3GB storage |
| **Railway** | ✅ $5 credit/month | ✅ Yes | Usually free with credit |
| **Google Cloud Run** | ✅ 2M requests/month | ✅ Yes | Auto-scaling, complex setup |
| **Replit** | ✅ Yes | ⚠️ Limited | Best for testing only |
| **PythonAnywhere** | ✅ Yes | ⚠️ Limited | 1 web app, 512MB disk |

### Database Hosting

| Service | Free Tier | Storage | Best For |
|---------|-----------|---------|----------|
| **Supabase** | ✅ Yes | 500MB | Easy setup, good docs |
| **Neon** | ✅ Yes | 3GB | Serverless, auto-scaling |
| **Railway** | ✅ Included | Varies | With Railway backend |
| **Render** | ✅ Yes | 90 days retention | With Render backend |
| **Fly.io** | ✅ Yes | 3GB | With Fly.io backend |
| **ElephantSQL** | ✅ Yes | 20MB | Very small projects only |

---

## 🚀 Quick Start: Completely Free Setup

### Step 1: Database (Supabase - 5 minutes)

1. Go to https://supabase.com/
2. Sign up (free)
3. Create new project
4. Wait for setup (2-3 minutes)
5. Go to SQL Editor
6. Run this SQL:
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
7. Copy connection string from Settings → Database

### Step 2: Backend (Render - 10 minutes)

1. Go to https://render.com/
2. Sign up (free)
3. New → Web Service
4. Connect GitHub repo
5. Settings:
   - **Name**: `facial-recognition-backend`
   - **Root Directory**: `sevices/face-recognition`
   - **Environment**: Docker
   - **Dockerfile Path**: `Dockerfile`
   - **Docker Context**: `sevices/face-recognition`
6. Environment Variables:
   ```env
   USE_DATABASE=true
   DATABASE_URL=<from-supabase>
   DB_HOST=<supabase-host>
   DB_PORT=5432
   DB_NAME=postgres
   DB_USER=postgres
   DB_PASSWORD=<supabase-password>
   DB_TABLE_NAME=visitors
   DB_VISITOR_ID_COLUMN=id
   DB_IMAGE_COLUMN=base64Image
   DB_VISITOR_LIMIT=0
   CORS_ORIGINS=https://your-app.vercel.app
   MODELS_PATH=/app/app/models
   ```
7. Deploy (takes 5-10 minutes)
8. Copy backend URL

### Step 3: Frontend (Vercel - 5 minutes)

1. Go to https://vercel.com/
2. Sign up with GitHub
3. Import repository
4. Settings:
   - **Root Directory**: `apps/facial_recog_web_app`
   - **Framework**: Next.js (auto-detected)
5. Environment Variables:
   ```env
   NEXT_PUBLIC_API_URL=https://your-backend.onrender.com
   DATABASE_URL=<supabase-connection-string>
   AUTH_SECRET=<generate-with-openssl-rand-base64-32>
   ```
6. Deploy

**Total Time**: ~20 minutes
**Total Cost**: $0/month

---

## 💡 Tips for Free Tiers

### Render (Sleeping Service)

**Problem**: Render free tier sleeps after 15 minutes of inactivity.

**Solutions**:
1. **Use Uptime Robot** (free): https://uptimerobot.com/
   - Set up monitoring to ping your service every 5 minutes
   - Keeps service awake

2. **Accept the delay**: First request after sleep takes 30-60 seconds
   - Add loading indicator in frontend
   - Users understand it's free

3. **Upgrade to paid**: $7/month for always-on (optional)

### Database Size Limits

**Supabase (500MB)**:
- ~50,000-100,000 visitors (depending on image size)
- Enough for most small-medium projects

**Neon (3GB)**:
- ~300,000-600,000 visitors
- Better for larger projects

**If you exceed limits**:
- Archive old data
- Use image compression
- Upgrade to paid tier (usually $5-10/month)

### Fly.io Free Tier

**Limits**:
- 3 shared VMs
- 3GB storage
- 160GB outbound data transfer

**Tips**:
- Monitor usage in dashboard
- Optimize Docker image size
- Use efficient ML models

---

## 🔄 Migration Between Free Services

### Moving from Render to Fly.io

1. Export environment variables from Render
2. Create Fly.io app
3. Set same environment variables
4. Deploy
5. Update Vercel `NEXT_PUBLIC_API_URL`
6. Delete Render service

### Moving from Supabase to Neon

1. Export data from Supabase:
   ```bash
   pg_dump $SUPABASE_URL > backup.sql
   ```
2. Create Neon project
3. Import data:
   ```bash
   psql $NEON_URL < backup.sql
   ```
4. Update backend `DATABASE_URL`
5. Restart backend service

---

## 📈 When to Upgrade

Consider upgrading to paid tiers when:

1. **Traffic**: >10,000 requests/day
2. **Database**: >500MB (Supabase) or >3GB (Neon)
3. **Uptime**: Need 99.9% uptime (Render sleep is issue)
4. **Performance**: Need faster response times
5. **Support**: Need priority support

**Typical Paid Costs**:
- Render: $7/month (always-on)
- Supabase: $25/month (8GB database)
- Neon: $19/month (10GB database)
- Fly.io: $1.94/month per VM

---

## 🎓 Learning Resources

- **Supabase Docs**: https://supabase.com/docs
- **Neon Docs**: https://neon.tech/docs
- **Render Docs**: https://render.com/docs
- **Fly.io Docs**: https://fly.io/docs
- **Vercel Docs**: https://vercel.com/docs

---

## ✅ Checklist: Free Deployment

- [ ] Choose free tier combination
- [ ] Set up database (Supabase/Neon)
- [ ] Initialize database schema
- [ ] Deploy backend (Render/Fly.io)
- [ ] Configure environment variables
- [ ] Deploy frontend (Vercel)
- [ ] Test all endpoints
- [ ] Set up uptime monitoring (if using Render)
- [ ] Monitor usage and limits
- [ ] Document your setup

---

## 🆘 Troubleshooting Free Tiers

### Render Service Sleeping

**Symptom**: First request takes 30-60 seconds

**Solution**: 
- Use Uptime Robot to keep awake
- Or upgrade to paid tier

### Database Connection Errors

**Symptom**: Can't connect to database

**Solution**:
- Check connection string format
- Verify database is running (Supabase/Neon always on)
- Check firewall/security settings
- Verify credentials

### Out of Storage

**Symptom**: Database full error

**Solution**:
- Archive old data
- Compress images
- Upgrade to paid tier
- Move to larger free tier (Neon has 3GB)

---

## 📝 Summary

**Best Free Stack for Most Users**:
- Frontend: **Vercel** (free, always-on)
- Backend: **Fly.io** (free, always-on)
- Database: **Neon** (free, 3GB)

**Total Cost**: $0/month
**Setup Time**: ~30 minutes
**Best For**: Production-ready applications

See [VERCEL_DEPLOYMENT.md](./VERCEL_DEPLOYMENT.md) for detailed step-by-step instructions.
