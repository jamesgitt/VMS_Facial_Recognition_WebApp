/**
 * Run `build` or `dev` with `SKIP_ENV_VALIDATION` to skip env validation. This is especially useful
 * for Docker builds.
 */
import "./src/env.js";

/** @type {import("next").NextConfig} */
const config = {
  // Use standalone output for Docker, but Vercel will use its own optimized output
  ...(process.env.SKIP_ENV_VALIDATION === "true" ? {} : {}),
  // Only use standalone for Docker builds, not Vercel
  ...(process.env.DOCKER_BUILD === "true" ? { output: 'standalone' } : {}),
};

export default config;
