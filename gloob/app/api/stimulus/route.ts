import { NextRequest } from "next/server";

function sanitizeHtml(input: string): string {
  let html = input;
  // Remove dangerous tags completely
  html = html.replace(/<\/(?:script|iframe|object|embed|frame|frameset|meta)[^>]*>/gi, "");
  html = html.replace(/<(?:script|iframe|object|embed|frame|frameset|meta)[^>]*>/gi, "");
  // Remove event handler attributes like onclick, onload, etc. (double/single/no quotes)
  html = html.replace(/\s(on[a-z]+)\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)/gi, "");
  // Neutralize javascript: URLs
  html = html.replace(/\s(href|src)\s*=\s*("|')\s*javascript:[^\2]*\2/gi, ' $1="#"');
  html = html.replace(/\s(href|src)\s*=\s*javascript:[^\s>]+/gi, ' $1="#"');
  // Remove external resource links; allow CSS only via <style>
  html = html.replace(/<link[^>]*>/gi, "");
  // Remove markdown fences if present
  html = html.replace(/```(?:html)?/gi, "");
  html = html.replace(/```/g, "");
  // Enforce read-only: strip common interactive elements
  html = html.replace(/<\/?(?:input|textarea|select|option|button)[^>]*>/gi, "");
  // Remove contenteditable attributes
  html = html.replace(/\scontenteditable\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)/gi, "");
  // Remove top-level document wrappers if present
  html = html.replace(/<!doctype[^>]*>/gi, "");
  html = html.replace(/<\/?(?:html|head|body|title)[^>]*>/gi, "");
  // Disallow forms to avoid unintended submissions
  html = html.replace(/<\/?form[^>]*>/gi, "");
  return html;
}

export async function POST(req: NextRequest) {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    return new Response("Missing OPENROUTER_API_KEY", { status: 500 });
  }

  const { scenario, language } = (await req.json().catch(() => ({}))) as {
    scenario?: string;
    language?: string;
  };

  if (!scenario || !scenario.trim()) {
    return new Response("Missing scenario", { status: 400 });
  }

  const baseSystemPrompt =
    "You generate a single self-contained HTML snippet with inline CSS for language practice. Output ONLY the HTML/CSS snippet. Do NOT include <html>, <head>, or <body> tags, and absolutely no <script> tags. No external resources or images. The snippet must be wrapped in a single <section id=\"stimulus\"> container with any required <style> inside it. Keep it minimal, readable, and visually clear. The stimulus must be strictly read-only: do NOT include forms, inputs, textareas, selects, options, buttons, toggles, sliders, or any interactive element (including contenteditable or links acting as buttons).";

  const upstream = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${apiKey}`,
      // Optional ranking metadata; set via env if provided
      ...(process.env.OPENROUTER_REFERER
        ? { "HTTP-Referer": process.env.OPENROUTER_REFERER }
        : {}),
      ...(process.env.OPENROUTER_TITLE
        ? { "X-Title": process.env.OPENROUTER_TITLE }
        : {}),
    },
    body: JSON.stringify({
      model: process.env.OPENROUTER_MODEL || "openai/gpt-5-mini",
      stream: false,
      messages: [
        { role: "system", content: baseSystemPrompt },
        {
          role: "user",
          content: [
            `Scenario: ${scenario.trim()}`,
            language && language.trim()
              ? `Learning language: ${language.trim()} (ALL visible text must be in ${language.trim()}. Do not include any other language.)`
              : `Learning language: not provided (choose a reasonable target language)`,
            "Create a minimal but realistic UI that looks like it belongs to this scenario.",
            "Requirements:",
            "- Wrap everything in <section id=\"stimulus\">…</section>.",
            "- Use inline <style> inside that <section>.",
            "- No scripts, no external images or fonts.",
            "- Use semantic HTML and subtle styling.",
            "- Include labels, prices, and layout if relevant.",
            "- Keep it concise and polished.",
            "- DO NOT provide hints, instructions, tips, or suggested phrases to the learner—only present the raw stimulus UI/text.",
            "- READ-ONLY: do NOT include <form>, <input>, <textarea>, <select>, <option>, <button>, contenteditable, or any interactive widgets.",
          ].join("\n"),
        },
      ],
    }),
  });

  if (!upstream.ok) {
    const text = await upstream.text().catch(() => "");
    return new Response(text || "Upstream error", { status: upstream.status || 500 });
  }

  type ChatResponse = {
    choices?: Array<{
      message?: { role?: string; content?: string };
      finish_reason?: string;
    }>;
  };
  const json = (await upstream.json().catch(() => ({}))) as ChatResponse;
  const raw = json?.choices?.[0]?.message?.content || "";
  const safe = sanitizeHtml(raw);

  return Response.json({ html: safe }, { status: 200, headers: { "Cache-Control": "no-store" } });
}
