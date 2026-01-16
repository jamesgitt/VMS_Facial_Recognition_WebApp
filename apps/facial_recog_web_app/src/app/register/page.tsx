"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";

export default function RegisterPage() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSuccess(false);
    setLoading(true);

    try {
      // Normalize email (lowercase and trim) to match registration API
      const normalizedEmail = email.toLowerCase().trim();
      
      const res = await fetch("/api/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ 
          email: normalizedEmail, 
          password, 
          name: name.trim() || undefined 
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        setError(data.error || "Registration failed");
        setLoading(false);
        return;
      }

      setSuccess(true);
      setLoading(false);
      
      // Redirect to sign in after 2 seconds
      setTimeout(() => {
        router.push("/signin?registered=true");
      }, 2000);
    } catch (err) {
      console.error("[Register] Registration error:", err);
      setError("Something went wrong. Please try again.");
      setLoading(false);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-b from-[#2e026d] to-[#15162c]">
      <div className="w-full max-w-md rounded-lg bg-white/10 p-8 shadow-lg">
        <h1 className="mb-6 text-center text-3xl font-bold text-white">
          Create Account
        </h1>
        
        {success ? (
          <div className="space-y-4">
            <div className="rounded-md bg-green-500/20 p-3 text-sm text-green-200">
              Account created successfully! Redirecting to sign in...
            </div>
          </div>
        ) : (
          <form onSubmit={handleSubmit} className="space-y-4">
            <div>
              <label htmlFor="name" className="block text-sm font-medium text-white">
                Name (Optional)
              </label>
              <input
                id="name"
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                className="mt-1 block w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-gray-900 focus:border-purple-500 focus:outline-none focus:ring-purple-500"
                placeholder="Your Name"
              />
            </div>

            <div>
              <label htmlFor="email" className="block text-sm font-medium text-white">
                Email
              </label>
              <input
                id="email"
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className="mt-1 block w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-gray-900 focus:border-purple-500 focus:outline-none focus:ring-purple-500"
                placeholder="your@email.com"
              />
            </div>

            <div>
              <label htmlFor="password" className="block text-sm font-medium text-white">
                Password
              </label>
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                minLength={6}
                className="mt-1 block w-full rounded-md border border-gray-300 bg-white px-3 py-2 text-gray-900 focus:border-purple-500 focus:outline-none focus:ring-purple-500"
                placeholder="•••••••• (min 6 characters)"
              />
              <p className="mt-1 text-xs text-white/70">
                Password must be at least 6 characters
              </p>
            </div>

            {error && (
              <div className="rounded-md bg-red-500/20 p-3 text-sm text-red-200">
                {error}
              </div>
            )}

            <button
              type="submit"
              disabled={loading}
              className="w-full rounded-md bg-purple-600 px-4 py-2 font-semibold text-white hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 disabled:opacity-50"
            >
              {loading ? "Creating account..." : "Create Account"}
            </button>
          </form>
        )}

        <p className="mt-4 text-center text-sm text-white/70">
          Already have an account?{" "}
          <a href="/signin" className="text-purple-300 hover:text-purple-200">
            Sign in here
          </a>
        </p>
      </div>
    </div>
  );
}
