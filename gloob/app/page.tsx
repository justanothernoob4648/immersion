"use client";

import { useRouter } from "next/navigation";
import { useCallback, useMemo, useRef, useState, useEffect } from "react";

type Language =
  | "English"
  | "Chinese"
  | "French"
  | "German"
  | "Japanese"
  | "Spanish";

type GridConfig = {
  cols: number;
  rows: number;
  originX: number;
  originY: number;
  baseDelay: number; // ms per tile distance
  duration: number; // ms per tile animation
};

export default function Home() {
  const router = useRouter();

  const languages: Language[] = useMemo(
    () => ["English", "Chinese", "French", "German", "Japanese", "Spanish"],
    []
  );

  const [scenario, setScenario] = useState("");
  const [nativeLang, setNativeLang] = useState<Language>("English");
  const [learnLang, setLearnLang] = useState<Language>("Spanish");
  const [proficiency, setProficiency] = useState<"beginner" | "intermediate" | "advanced">(
    "beginner"
  );
  const [animating, setAnimating] = useState(false);
  const [grid, setGrid] = useState<GridConfig | null>(null);
  const buttonRef = useRef<HTMLButtonElement | null>(null);
  const [waiting, setWaiting] = useState(false);
  // no extra overlay states (reverted to simple tile animation)

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      if (!scenario.trim()) return;
      const overlayMinMs = 700; // ensure user perceives the overlay
      const startTs = performance.now();
      // Compute a responsive grid based on viewport size
      const w = window.innerWidth;
      const h = window.innerHeight;
      const approxTile = 64; // px target tile size for crisper mosaic
      const cols = Math.max(8, Math.ceil(w / approxTile));
      const rows = Math.max(6, Math.ceil(h / approxTile));

      // Find wave origin near the submit button center
      const rect = buttonRef.current?.getBoundingClientRect();
      const cx = rect ? rect.left + rect.width / 2 : w / 2;
      const cy = rect ? rect.top + rect.height / 2 : h / 2;
      const originX = Math.min(cols - 1, Math.max(0, Math.floor((cx / w) * cols)));
      const originY = Math.min(rows - 1, Math.max(0, Math.floor((cy / h) * rows)));

      const baseDelay = 18; // retained for wave-origin visuals if needed
      const duration = 520;
      setGrid({ cols, rows, originX, originY, baseDelay, duration });
      setAnimating(true);
      setWaiting(true);

      // Simple tile overlay uses only grid sizing

      // Request the full HTML stimulus before navigating
      try {
        const res = await fetch("/api/stimulus", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ scenario: scenario.trim(), language: learnLang }),
        });
        if (!res.ok) {
          const t = await res.text().catch(() => "");
          throw new Error(t || `Failed (${res.status})`);
        }
        const data = (await res.json()) as { html?: string };
        const html = data.html || "";
        // Persist to sessionStorage under a unique key
        const key = crypto?.randomUUID?.() || String(Date.now());
        try {
          sessionStorage.setItem(`stimulus:${key}`, html);
        } catch {}

        // Create a new server session with metadata
        let sessionId = "";
        try {
          const startRes = await fetch("/api/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              scenario: scenario.trim(),
              native_language: nativeLang,
              target_language: learnLang,
              proficiency,
              stimulusHtml: html,
            }),
          });
          if (startRes.ok) {
            const sj = (await startRes.json()) as { sessionId?: string };
            sessionId = sj.sessionId || "";
          }
        } catch {}

        // Ensure the overlay has been visible for at least overlayMinMs
        const elapsed = performance.now() - startTs;
        if (elapsed < overlayMinMs) {
          await new Promise((r) => setTimeout(r, overlayMinMs - elapsed));
        }

        // Navigate only after we have the full result
        const url = new URL(window.location.origin + "/learn");
        url.searchParams.set("scenario", scenario.trim());
        url.searchParams.set("k", key);
        url.searchParams.set("native", nativeLang);
        url.searchParams.set("learn", learnLang);
        if (sessionId) url.searchParams.set("session", sessionId);
        router.push(url.pathname + url.search);
      } catch (err) {
        console.error(err);
        // Gracefully stop animation but stay on page; surface minimal feedback
        alert("Sorry, failed to generate the stimulus. Please try again.");
        setAnimating(false);
        setWaiting(false);
      }
    },
    [router, scenario, nativeLang, learnLang, proficiency]
  );

  // No extra effects needed for simple tile animation

  return (
    <div className="relative min-h-screen overflow-hidden bg-[var(--background)] text-[var(--foreground)]">
      {/* Background overlays */}
      <div aria-hidden className="pointer-events-none absolute inset-0 z-0">
        <div className="absolute inset-0 bg-radial" />
        <div className="absolute inset-0 bg-grid" />
        <div className="absolute inset-0 bg-grain" />
      </div>

      <main className="relative z-10 mx-auto flex min-h-screen w-full max-w-2xl items-center justify-center px-6 py-16">
        <div className="w-full rounded-2xl border border-[var(--border)] bg-[var(--surface)]/90 p-8 shadow-sm">
          <header className="mb-8">
            <h1 className="text-center text-4xl font-semibold tracking-tight">
              Start Speaking
            </h1>
            <p className="mt-2 text-center text-sm" style={{ color: "var(--text-muted)" }}>
              Supports any scenario you desire.
            </p>
          </header>

          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label htmlFor="scenario" className="block text-sm font-medium">
                Scenario
              </label>
              <textarea
                id="scenario"
                value={scenario}
                onChange={(e) => setScenario(e.target.value)}
                placeholder="Ordering coffee, returning an item, asking directions..."
                className="mt-2 h-32 w-full resize-none rounded-xl border border-[var(--border)] bg-[var(--muted)]/60 px-4 py-3 text-base outline-none transition focus:border-[var(--accent)]"
              />
            </div>

            <div className="grid gap-4 sm:grid-cols-2">
              <div>
                <label htmlFor="native" className="block text-sm font-medium">
                  Native language
                </label>
                <div className="relative mt-2">
                  <select
                    id="native"
                    value={nativeLang}
                    onChange={(e) => setNativeLang(e.target.value as Language)}
                    className="w-full appearance-none rounded-xl border border-[var(--border)] bg-[var(--muted)]/60 px-4 py-3 text-base outline-none transition focus:border-[var(--accent)]"
                  >
                    {languages.map((lang) => (
                      <option key={lang} value={lang}>
                        {lang}
                      </option>
                    ))}
                  </select>
                  <span className="pointer-events-none absolute right-4 top-1/2 -translate-y-1/2 text-zinc-500">▾</span>
                </div>
              </div>
              <div>
                <label htmlFor="learn" className="block text-sm font-medium">
                  Learning language
                </label>
                <div className="relative mt-2">
                  <select
                    id="learn"
                    value={learnLang}
                    onChange={(e) => setLearnLang(e.target.value as Language)}
                    className="w-full appearance-none rounded-xl border border-[var(--border)] bg-[var(--muted)]/60 px-4 py-3 text-base outline-none transition focus:border-[var(--accent)]"
                  >
                    {languages.map((lang) => (
                      <option key={lang} value={lang}>
                        {lang}
                      </option>
                    ))}
                  </select>
                  <span className="pointer-events-none absolute right-4 top-1/2 -translate-y-1/2 text-zinc-500">▾</span>
                </div>
              </div>
            </div>

            <div>
              <label htmlFor="proficiency" className="block text-sm font-medium">
                Proficiency
              </label>
              <div className="relative mt-2">
                <select
                  id="proficiency"
                  value={proficiency}
                  onChange={(e) =>
                    setProficiency(e.target.value as "beginner" | "intermediate" | "advanced")
                  }
                  className="w-full appearance-none rounded-xl border border-[var(--border)] bg-[var(--muted)]/60 px-4 py-3 text-base outline-none transition focus:border-[var(--accent)]"
                >
                  <option value="beginner">Beginner</option>
                  <option value="intermediate">Intermediate</option>
                  <option value="advanced">Advanced</option>
                </select>
                <span className="pointer-events-none absolute right-4 top-1/2 -translate-y-1/2 text-zinc-500">▾</span>
              </div>
            </div>

            {nativeLang === learnLang && (
              <p className="text-sm" style={{ color: "var(--text-muted)" }}>
                Tip: Choosing different languages is usually more helpful.
              </p>
            )}

            <div className="pt-2">
              <button
                ref={buttonRef}
                type="submit"
                disabled={!scenario.trim() || animating || waiting}
                className="relative inline-flex w-full items-center justify-center gap-2 rounded-xl bg-[var(--accent)] px-6 py-3 text-lg font-semibold text-white shadow-sm transition-colors hover:bg-[var(--accent-600)] active:bg-[var(--accent-700)] disabled:cursor-not-allowed"
              >
                {animating || waiting ? (
                  <>
                    <span className="h-5 w-5 animate-spin rounded-full border-2 border-white/60 border-t-transparent" />
                    Preparing practice…
                  </>
                ) : (
                  <>Begin Practice</>
                )}
              </button>
            </div>
          </form>
        </div>
      </main>

      {/* Creative tile transition overlay */}
      {animating && grid && (
        <div
          aria-hidden
          className="pointer-events-auto fixed inset-0 z-[9999] grid"
          style={{
            background: "#ffffff",
            gridTemplateColumns: `repeat(${grid.cols}, 1fr)`,
            gridTemplateRows: `repeat(${grid.rows}, 1fr)`,
          }}
        >
          {Array.from({ length: grid.cols * grid.rows }).map((_, i) => {
            const x = i % grid.cols;
            const y = Math.floor(i / grid.cols);
            const d = Math.hypot(x - grid.originX, y - grid.originY);
            const delay = Math.round(d * grid.baseDelay);
            const pulseDelay = delay + grid.duration + Math.round(Math.random() * 1200);
            const pulseDur = Math.round(1600 + Math.random() * 1400);
            const palette = [
              "var(--tile-1)",
              "var(--tile-2)",
              "var(--tile-3)",
              "var(--tile-4)",
              "var(--tile-5)",
            ];
            const bg = palette[(x + y) % palette.length];
            return (
              <div
                key={i}
                className="tile-anim"
                style={{
                  background: bg,
                  transformOrigin: y < grid.originY ? "top" : "bottom",
                  animation: `tileGrow ${grid.duration}ms cubic-bezier(0.22,1,0.36,1) ${delay}ms both, tilePulse ${pulseDur}ms ease-in-out ${pulseDelay}ms infinite alternate`,
                  willChange: "filter, transform, opacity",
                }}
              />
            );
          })}
        </div>
      )}
    </div>
  );
}
