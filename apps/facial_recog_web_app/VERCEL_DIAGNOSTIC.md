# Vercel 404 Diagnostic Steps

## Immediate Actions

### 1. Check Build Logs in Vercel

**This is the MOST IMPORTANT step:**

1. Go to Vercel Dashboard
2. Click on your project
3. Go to "Deployments" tab
4. Click on the failed deployment (the one with 404)
5. Click "Build Logs" tab
6. **Scroll through the entire log** and look for:
   - ❌ Red error messages
   - ⚠️ Yellow warnings
   - Any mention of "failed", "error", "missing"

**Common errors to look for:**
- `Invalid environment variables`
- `Module not found`
- `Prisma Client not generated`
- `Type error`
- `Build failed`

**Copy the error message and share it** - this will tell us exactly what's wrong.

---

### 2. Verify Root Directory

1. Vercel Dashboard → Your Project
2. Settings → General
3. Scroll to "Root Directory"
4. **Must be exactly**: `apps/facial_recog_web_app`
5. If it's wrong, change it and click "Save"
6. Redeploy

---

### 3. Check Environment Variables

Go to Settings → Environment Variables and verify ALL of these exist:

**Required:**
- ✅ `SKIP_ENV_VALIDATION` = `true` (CRITICAL!)
- ✅ `DATABASE_URL` = `postgresql://...`
- ✅ `AUTH_SECRET` = `...` (any string, can be generated)
- ✅ `NEXT_PUBLIC_API_URL` = `https://...` (your backend URL)

**Make sure:**
- All variables are set for **all environments** (Production, Preview, Development)
- No typos in variable names
- Values don't have extra spaces

---

### 4. Test Build Locally

Before deploying, test if it builds locally:

```bash
# Navigate to frontend directory
cd apps/facial_recog_web_app

# Install dependencies
npm install

# Set environment variables (create .env.local file)
echo "SKIP_ENV_VALIDATION=true" > .env.local
echo "DATABASE_URL=postgresql://user:pass@localhost:5432/db" >> .env.local
echo "AUTH_SECRET=test-secret" >> .env.local
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" >> .env.local

# Try to build
npm run build
```

**If local build fails**, fix those errors first before deploying to Vercel.

---

### 5. Check Vercel Project Settings

1. Go to Settings → General
2. Verify:
   - **Framework Preset**: Next.js
   - **Build Command**: `npm run build` (or empty for auto)
   - **Output Directory**: `.next` (or empty for auto)
   - **Install Command**: `npm install` (or empty for auto)
   - **Node Version**: 20.x (recommended)

---

### 6. Clear Build Cache

1. Settings → General
2. Scroll to "Build & Development Settings"
3. Click "Clear Build Cache"
4. Redeploy

---

## Common Issues and Fixes

### Issue: "Invalid environment variables"

**Fix:**
- Add `SKIP_ENV_VALIDATION=true` to environment variables
- Make sure it's set for all environments

---

### Issue: "Prisma Client not generated"

**Fix:**
- Check `package.json` has: `"postinstall": "prisma generate"`
- Make sure `prisma/schema.prisma` exists
- Try adding to build command: `npm run postinstall && npm run build`

---

### Issue: "Module not found"

**Fix:**
- Run `npm install` locally
- Make sure all dependencies are in `package.json`
- Check if `node_modules` is in `.gitignore` (it should be)

---

### Issue: Build succeeds but still 404

**Fix:**
- Check Root Directory is correct
- Verify `src/app/page.tsx` exists
- Check if `next.config.js` has any redirects/rewrites that might interfere
- Try accessing different URLs:
  - `https://your-app.vercel.app/` (homepage)
  - `https://your-app.vercel.app/camera` (camera page)
  - `https://your-app.vercel.app/api/trpc` (API route)

---

## Quick Fix Checklist

Run through this checklist:

- [ ] Root Directory = `apps/facial_recog_web_app`
- [ ] `SKIP_ENV_VALIDATION=true` is set
- [ ] All required environment variables are set
- [ ] Build logs checked for errors
- [ ] Local build tested (`npm run build`)
- [ ] Build cache cleared
- [ ] Redeployed after fixes

---

## Still Not Working?

If you've done all the above and still get 404:

1. **Share the Build Logs** - Copy the error from Vercel build logs
2. **Share your Vercel settings** - Screenshot of Root Directory and Build Settings
3. **Test local build** - Share the output of `npm run build` locally

The build logs will tell us exactly what's failing!
