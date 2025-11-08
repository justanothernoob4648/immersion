import { NextRequest } from "next/server";

const getServerOrigin = () =>
  process.env.PIPECAT_SERVER_ORIGIN?.replace(/\/$/, "") || "http://localhost:7860";

export async function POST(req: NextRequest) {
  const origin = getServerOrigin();
  const url = `${origin}/start`;
  try {
    const body = await req.json();
    const upstream = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const text = await upstream.text();
    return new Response(text, {
      status: upstream.status,
      headers: { "Content-Type": upstream.headers.get("content-type") || "application/json" },
    });
  } catch (e: any) {
    return new Response(e?.message || "Proxy error", { status: 500 });
  }
}

