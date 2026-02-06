import Link from "next/link";

import { LatestPost } from "~/app/_components/post";
import { auth } from "~/server/auth";
import { api, HydrateClient } from "~/trpc/server";

export default async function Home() {
  // Wrap in try-catch to prevent build failures
  let hello = null;
  let session = null;
  
  try {
    hello = await api.post.hello({ text: "from tRPC" });
  } catch (error) {
    console.error("tRPC error:", error);
  }
  
  try {
    session = await auth();
    if (session?.user) {
      void api.post.getLatest.prefetch();
    }
  } catch (error) {
    console.error("Auth error:", error);
  }

  return (
    <HydrateClient>
      <main className="flex min-h-screen flex-col items-center justify-center bg-white">
        <div className="container flex flex-col items-center justify-center gap-12 px-4 py-16">
          {/* Header */}
          <div className="text-center">
            <h1 className="text-5xl font-extrabold tracking-tight text-[#001738] sm:text-[4rem]">
              VMS <span className="text-[#ff7200]">Face Recognition</span>
            </h1>
            <p className="mt-4 text-lg text-[#001738]/70">
              Visitor Management System with AI-powered face recognition
            </p>
          </div>

          {/* Cards */}
          <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 md:gap-8">
            <Link
              className="flex max-w-sm flex-col gap-4 rounded-xl border-2 border-[#001738]/10 bg-white p-6 shadow-lg transition-all hover:border-[#ff7200] hover:shadow-xl"
              href="/camera"
            >
              <div className="flex items-center gap-3">
                <div className="rounded-full bg-[#ff7200] p-3">
                  <svg className="h-6 w-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                  </svg>
                </div>
                <h3 className="text-2xl font-bold text-[#001738]">Face Recognition</h3>
              </div>
              <p className="text-[#001738]/70">
                Real-time face detection and recognition using your webcam and AI.
              </p>
              <span className="text-[#ff7200] font-semibold">Open Camera →</span>
            </Link>

            <Link
              className="flex max-w-sm flex-col gap-4 rounded-xl border-2 border-[#001738]/10 bg-white p-6 shadow-lg transition-all hover:border-[#ff7200] hover:shadow-xl"
              href="/register"
            >
              <div className="flex items-center gap-3">
                <div className="rounded-full bg-[#001738] p-3">
                  <svg className="h-6 w-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M18 9v3m0 0v3m0-3h3m-3 0h-3m-2-5a4 4 0 11-8 0 4 4 0 018 0zM3 20a6 6 0 0112 0v1H3v-1z" />
                  </svg>
                </div>
                <h3 className="text-2xl font-bold text-[#001738]">Register Visitor</h3>
              </div>
              <p className="text-[#001738]/70">
                Add a new visitor to the system with their photo and details.
              </p>
              <span className="text-[#ff7200] font-semibold">Register →</span>
            </Link>
          </div>

          {/* Status */}
          <div className="flex flex-col items-center gap-4">
            <p className="text-lg text-[#001738]/60">
              {hello ? hello.greeting : "Connecting to server..."}
            </p>

            <Link
              href="/signin"
              className="rounded-full bg-[#001738] px-8 py-3 font-semibold text-white transition hover:bg-[#002855]"
            >
              Sign In
            </Link>
          </div>

          {session?.user && <LatestPost />}
        </div>
      </main>
    </HydrateClient>
  );
}
