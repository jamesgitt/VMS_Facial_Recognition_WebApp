# Vercel Deployment Troubleshooting

## Common Issues and Solutions

### 404: NOT_FOUND Error

This error typically occurs when Vercel can't find or build your application correctly.

#### Solution 1: Check Root Directory

**In Vercel Dashboard:**
1. Go to your project → Settings → General
2. Scroll to "Root Directory"
3. Set it to: `apps/facial_recog_web_app`
4. Click "Save"
5. Redeploy

**Or via CLI:**
```bash
vercel --cwd apps/facial_recog_web_app
```

---

#### Solution 2: Verify Build Configuration

**In Vercel Dashboard:**
1. Go to Settings → General
2. Verify:
   - **Framework Preset**: Next.js
   - **Build Command**: `npm run build` (or leave empty for auto-detection)
   - **Output Directory**: `.next` (or leave empty for auto-detection)
   - **Install Command**: `npm install` (or leave empty for auto-detection)

---

#### Solution 3: Check Environment Variables

**Required Environment Variables:**
```env
# API URL (from your backend deployment)
NEXT_PUBLIC_API_URL=https://your-backend.railway.app

# Database (for Prisma)
DATABASE_URL=postgresql://user:password@host:5432/visitors_db

# Authentication (generate with: openssl rand -base64 32)
AUTH_SECRET=your-generated-secret

# Skip environment validation during build
SKIP_ENV_VALIDATION=true
```

**To add in Vercel:**
1. Go to Settings → Environment Variables
2. Add each variable
3. Make sure to select all environments (Production, Preview, Development)
4. Redeploy

---

#### Solution 4: Check Build Logs

1. Go to your deployment in Vercel Dashboard
2. Click on the failed deployment
3. Check "Build Logs" tab
4. Look for errors like:
   - Missing dependencies
   - TypeScript errors
   - Environment variable validation errors
   - Prisma generation errors

**Common Build Errors:**

**Error: "Invalid environment variables"**
- Solution: Add `SKIP_ENV_VALIDATION=true` to environment variables

**Error: "Prisma Client not generated"**
- Solution: The `postinstall` script should run `prisma generate` automatically
- If not, add to `package.json`:
  ```json
  "scripts": {
    "postinstall": "prisma generate"
  }
  ```

**Error: "Module not found"**
- Solution: Make sure all dependencies are in `package.json`
- Run `npm install` locally to verify

---

#### Solution 5: Simplify vercel.json

The `vercel.json` file should be minimal. Current version:
```json
{
  "framework": "nextjs",
  "regions": ["iad1"]
}
```

Vercel auto-detects Next.js, so you can even remove `vercel.json` entirely if needed.

---

#### Solution 6: Check File Structure

Make sure your project structure is:
```
apps/facial_recog_web_app/
├── src/
│   └── app/
│       ├── layout.tsx
│       ├── page.tsx
│       └── ...
├── package.json
├── next.config.js
└── vercel.json
```

---

#### Solution 7: Deploy from Correct Directory

**If deploying via CLI:**
```bash
# Navigate to frontend directory
cd apps/facial_recog_web_app

# Deploy
vercel

# Or specify root explicitly
vercel --cwd apps/facial_recog_web_app
```

**If deploying via GitHub:**
- Make sure Vercel is connected to your GitHub repo
- Set Root Directory in Vercel dashboard to `apps/facial_recog_web_app`

---

#### Solution 8: Clear Build Cache

1. Go to Settings → General
2. Scroll to "Build & Development Settings"
3. Click "Clear Build Cache"
4. Redeploy

---

#### Solution 9: Check Next.js Version Compatibility

Make sure you're using a compatible Next.js version:
```json
{
  "dependencies": {
    "next": "^15.2.3"
  }
}
```

Vercel supports Next.js 15, but if you have issues, try:
```bash
npm install next@latest
```

---

#### Solution 10: Verify Prisma Setup

**Make sure Prisma schema exists:**
- File: `apps/facial_recog_web_app/prisma/schema.prisma`

**Make sure postinstall script runs:**
- Check `package.json` has: `"postinstall": "prisma generate"`

**If Prisma fails:**
- Add to environment variables: `SKIP_ENV_VALIDATION=true`
- Or modify `next.config.js` to skip env validation during build

---

## Step-by-Step Fix for 404 Error

1. **Verify Root Directory**:
   - Vercel Dashboard → Settings → General
   - Root Directory: `apps/facial_recog_web_app`
   - Save

2. **Add Environment Variables**:
   - Settings → Environment Variables
   - Add all required variables (see Solution 3)
   - Make sure `SKIP_ENV_VALIDATION=true` is set

3. **Check Build Logs**:
   - Go to failed deployment
   - Check Build Logs tab
   - Fix any errors shown

4. **Simplify Configuration**:
   - Use minimal `vercel.json` (or remove it)
   - Let Vercel auto-detect Next.js

5. **Redeploy**:
   - Trigger new deployment
   - Or push a commit to trigger auto-deploy

---

## Quick Test

After fixing, test your deployment:

1. **Check Homepage**:
   ```
   https://your-app.vercel.app
   ```
   Should show the main page

2. **Check API Routes**:
   ```
   https://your-app.vercel.app/api/trpc
   ```
   Should not return 404

3. **Check Camera Page**:
   ```
   https://your-app.vercel.app/camera
   ```
   Should show camera interface

---

## Still Having Issues?

1. **Check Vercel Status**: https://vercel-status.com/
2. **Review Build Logs**: Look for specific error messages
3. **Test Locally**: Run `npm run build` locally to catch errors
4. **Check GitHub Actions**: If using GitHub, check if commits are triggering deployments
5. **Contact Support**: Vercel support is helpful for deployment issues

---

## Common Error Messages

| Error | Solution |
|------|----------|
| `404: NOT_FOUND` | Check root directory, verify build succeeded |
| `Module not found` | Check dependencies, run `npm install` |
| `Invalid environment variables` | Add `SKIP_ENV_VALIDATION=true` |
| `Prisma Client not generated` | Check `postinstall` script in package.json |
| `Build failed` | Check build logs for specific error |
| `Deployment timeout` | Increase build timeout in settings |

---

## Prevention

To avoid 404 errors in the future:

1. ✅ Always set Root Directory correctly
2. ✅ Add all required environment variables before first deploy
3. ✅ Test build locally: `npm run build`
4. ✅ Keep `vercel.json` minimal (or remove it)
5. ✅ Monitor build logs for warnings
6. ✅ Use Vercel's auto-detection when possible
