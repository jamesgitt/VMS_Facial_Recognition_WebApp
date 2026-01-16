FROM node:20-alpine AS base

# Install dependencies only when needed
FROM base AS deps
RUN apk add --no-cache libc6-compat
WORKDIR /app

# Copy package files and Prisma schema (needed for postinstall script)
COPY package.json package-lock.json* ./
COPY prisma ./prisma
RUN npm ci

# Rebuild the source code only when needed
FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules

# Copy source files in separate layers to better handle cache invalidation
# Copy configuration files first (these change less frequently)
COPY package.json package-lock.json* ./
COPY next.config.js* ./
COPY tsconfig.json* ./
COPY postcss.config.js* ./
COPY eslint.config.js* ./
COPY prettier.config.js* ./
COPY tailwind.config.ts* ./

# Copy Prisma schema
COPY prisma ./prisma

# Copy source code (this will invalidate cache when source changes)
COPY src ./src
COPY public ./public

# Set environment variables for build
ENV NEXT_TELEMETRY_DISABLED=1
ENV NODE_ENV=production
ENV SKIP_ENV_VALIDATION=true
ENV DOCKER_BUILD=true

# Clean any existing build artifacts
RUN rm -rf .next

# Generate Prisma Client with correct binary targets for Alpine Linux
RUN npx prisma generate

# Build Next.js (this will create .next/standalone with all necessary files)
RUN npm run build

# Verify signin page was built (for debugging)
RUN ls -la .next/server/app/signin/ || echo "Warning: signin page not found in build"

# Production image, copy all the files and run next
FROM base AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# Copy necessary files from standalone build
# The standalone output includes a minimal server with all necessary files
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
COPY --from=builder --chown=nextjs:nodejs /app/generated ./generated

USER nextjs

EXPOSE 3000

ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["node", "server.js"]
