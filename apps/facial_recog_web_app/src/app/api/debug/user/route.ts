import { NextResponse } from "next/server";
import { db } from "~/server/db";

// Debug endpoint to check if a user exists
// Usage: GET /api/debug/user?email=test@example.com
export async function GET(request: Request) {
  try {
    const { searchParams } = new URL(request.url);
    const email = searchParams.get("email");

    if (!email) {
      return NextResponse.json(
        { error: "Email parameter is required" },
        { status: 400 }
      );
    }

    const normalizedEmail = email.toLowerCase().trim();
    const user = await db.user.findUnique({
      where: { email: normalizedEmail },
      select: {
        id: true,
        email: true,
        name: true,
        password: true, // Include password to check if it exists
      },
    });

    if (!user) {
      return NextResponse.json({
        exists: false,
        message: `User with email "${normalizedEmail}" not found`,
      });
    }

    return NextResponse.json({
      exists: true,
      user: {
        id: user.id,
        email: user.email,
        name: user.name,
        hasPassword: !!user.password,
        passwordLength: user.password?.length ?? 0,
      },
    });
  } catch (error) {
    console.error("[Debug] Error checking user:", error);
    return NextResponse.json(
      { error: "Internal server error", details: String(error) },
      { status: 500 }
    );
  }
}
