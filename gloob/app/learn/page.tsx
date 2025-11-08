"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

type GridConfig = {
  cols: number;
  rows: number;
  originX: number;
  originY: number;
  baseDelay: number;
  duration: number;
};

export default function LearnPage() {
  const router = useRouter();
  const search = useSearchParams();
  const scenario = useMemo(() => search.get("scenario") || "", [search]);
  const k = useMemo(() => search.get("k") || "", [search]);
  const sessionId = useMemo(() => search.get("session") || "", [search]);
  const learnLang = useMemo(() => search.get("learn") || "", [search]);
  const [html, setHtml] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [webrtcReady, setWebrtcReady] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState("Disconnected");
  const [messages, setMessages] = useState<Array<{ role: "user" | "assistant"; text: string; ts: number }>>([]);
  const botBufferRef = useRef<string>("");
  const finishBtnRef = useRef<HTMLButtonElement | null>(null);
  const [finishing, setFinishing] = useState(false);
  const [showConfetti, setShowConfetti] = useState(false);
  const [animating, setAnimating] = useState(false);
  const [grid, setGrid] = useState<GridConfig | null>(null);
  const [showEndModal, setShowEndModal] = useState(false);
  const [animKey, setAnimKey] = useState(0);

  useEffect(() => {
    try {
      const cached = sessionStorage.getItem(`stimulus:${k}`) || "";
      if (!cached) {
        setError("No prepared stimulus found. Please try again.");
      } else {
        setHtml(cached);
      }
    } catch {
      setError("Could not access prepared content.");
    }
  }, [k]);

  // Initialize the Pipecat WebRTC client once on mount
  useEffect(() => {
    let stopped = false;
    let client: any | null = null;
    let transport: any | null = null;

    const init = async () => {
      try {
        // Use bundler-aware dynamic imports so Next can rewrite module specifiers
        const [clientMod, transportMod] = await Promise.all([
          import("@pipecat-ai/client-js"),
          import("@pipecat-ai/small-webrtc-transport"),
        ]);
        const { PipecatClient } = clientMod as any;
        const { SmallWebRTCTransport } = transportMod as any;

        if (stopped) return;

        const $ = (id: string) => document.getElementById(id) as HTMLElement | null;

        const connectBtn = $("connect-btn") as HTMLButtonElement | null;
        const micBtn = $("mute-mic") as HTMLButtonElement | null;
        const camBtn = $("mute-btn") as HTMLButtonElement | null;

        const audioInput = $("audio-input") as HTMLSelectElement | null;
        const videoInput = $("video-input") as HTMLSelectElement | null;
        const audioCodec = $("audio-codec") as HTMLSelectElement | null;
        const videoCodec = $("video-codec") as HTMLSelectElement | null;

        const botVideo = $("bot-video") as HTMLVideoElement | null;
        const botAudio = $("bot-audio") as HTMLAudioElement | null;
        const localCam = $("local-cam") as HTMLVideoElement | null;

        const debugLog = $("debug-log");
        const statusSpan = $("connection-status");

        const log = (message: string) => {
          if (!debugLog) return;
          const entry = document.createElement("div");
          entry.textContent = `${new Date().toISOString()} - ${message}`;
          debugLog.appendChild(entry);
          debugLog.scrollTop = debugLog.scrollHeight;
        };

        const updateStatus = (status: string) => {
          setConnectionStatus(status);
          if (statusSpan) statusSpan.textContent = status;
          log(`Status: ${status}`);
        };

        const opts: any = {
          transport: new SmallWebRTCTransport({
            webrtcUrl: sessionId ? `/api/sessions/${sessionId}/offer` : "/api/offer",
          }),
          enableMic: true,
          enableCam: true,
          callbacks: {
            onTransportStateChanged: (state: any) => log(`Transport state: ${state}`),
            onConnected: () => {
              updateStatus("Connected");
              if (connectBtn) connectBtn.disabled = true;
            },
            onBotReady: () => log("Bot is ready."),
            onDisconnected: () => {
              updateStatus("Disconnected");
              if (connectBtn) connectBtn.disabled = false;
            },
            onUserStartedSpeaking: () => log("User started speaking."),
            onUserStoppedSpeaking: () => log("User stopped speaking."),
            onBotStartedSpeaking: () => log("Bot started speaking."),
            onUserTranscript: (transcript: any) => {
              if (transcript?.final) {
                log(`User transcript: ${transcript.text}`);
                setMessages((prev) => [
                  ...prev,
                  { role: "user", text: String(transcript.text || ""), ts: Date.now() },
                ]);
              }
            },
            onBotTranscript: (data: any) => {
              log(`Bot transcript: ${data.text}`);
              botBufferRef.current += String(data?.text || "");
            },
            onTrackStarted: (track: MediaStreamTrack, participant?: any) => {
              if (participant?.local) {
                // Local track
                if (track.kind === "audio") return;
                if (localCam) localCam.srcObject = new MediaStream([track]);
              } else {
                // Bot track
                if (track.kind === "video") {
                  if (botVideo) botVideo.srcObject = new MediaStream([track]);
                } else if (track.kind === "audio") {
                  if (botAudio) botAudio.srcObject = new MediaStream([track]);
                }
              }
            },
            onBotStoppedSpeaking: () => {
              const text = botBufferRef.current.trim();
              if (text) {
                setMessages((prev) => [...prev, { role: "assistant", text, ts: Date.now() }]);
                botBufferRef.current = "";
              }
            },
            onTrackStopped: (track: MediaStreamTrack, participant?: any) => {
              if (participant?.local) {
                if (track.kind === "video") {
                  if (localCam) localCam.srcObject = null;
                }
              }
            },
            onServerMessage: (msg: unknown) => log(`Server message: ${String(msg)}`),
          },
        };

        client = new PipecatClient(opts);
        transport = client.transport;

        // Populate devices
        const populateSelect = (sel: HTMLSelectElement | null, devices: MediaDeviceInfo[]) => {
          if (!sel) return;
          // clear previous
          sel.options.length = 0;
          const def = document.createElement("option");
          def.value = "";
          def.selected = true;
          def.text = "Default device";
          sel.appendChild(def);
          let counter = 1;
          devices.forEach((d) => {
            const opt = document.createElement("option");
            opt.value = d.deviceId;
            opt.text = d.label || `Device #${counter++}`;
            sel.appendChild(opt);
          });
        };
        try {
          const audioDevices = await client.getAllMics();
          const videoDevices = await client.getAllCams();
          populateSelect(audioInput, audioDevices);
          populateSelect(videoInput, videoDevices);
        } catch (e) {
          log(String(e));
        }

        // Wire events
        connectBtn?.addEventListener("click", async () => {
          debugLog && (debugLog.innerHTML = "");
          connectBtn.disabled = true;
          updateStatus("Connecting");
          try {
            if (audioCodec && typeof (transport as any)?.setAudioCodec === "function")
              (transport as any).setAudioCodec(audioCodec.value);
            if (videoCodec && typeof (transport as any)?.setVideoCodec === "function")
              (transport as any).setVideoCodec(videoCodec.value);
            await client!.connect();
          } catch (e) {
            log(`Failed to connect ${e}`);
            await client!.disconnect();
          }
        });

        // Removed disconnect button/UI; rely on auto-disconnect or page navigation

        audioInput?.addEventListener("change", (e: any) => {
          client!.updateMic(e?.target?.value);
        });

        micBtn?.addEventListener("click", async () => {
          if ((client as any).state === "disconnected") await client!.initDevices();
          else client!.enableMic(!client!.isMicEnabled);
        });

        videoInput?.addEventListener("change", (e: any) => {
          client!.updateCam(e?.target?.value);
        });

        camBtn?.addEventListener("click", async () => {
          if ((client as any).state === "disconnected") await client!.initDevices();
          else client!.enableCam(!client!.isCamEnabled);
        });

        // Screen share is removed from UI

        setWebrtcReady(true);
      } catch (err) {
        // Dependencies not installed or runtime cannot import
        console.error("Failed to initialize WebRTC client:", err);
      }
    };

    init();

    return () => {
      stopped = true;
      try {
        client?.disconnect?.();
      } catch {}
    };
  }, [sessionId]);

  return (
    <main className="relative min-h-screen bg-[var(--background)] text-[var(--foreground)]">
      <div className="mx-auto max-w-3xl px-6 py-10">
        <header className="mb-6">
          <h1 className="text-2xl font-semibold">Practice</h1>
          <p className="mt-1 text-sm" style={{ color: "var(--text-muted)" }}>
            Your practice stimulus is ready.
          </p>
        </header>

        <section className="mb-6 rounded-xl border border-[var(--border)] bg-[var(--surface)]/70 p-4">
          <h2 className="mb-2 text-sm font-medium">Scenario</h2>
          <p className="whitespace-pre-wrap text-sm" style={{ color: "var(--text-muted)" }}>
            {scenario || "(none)"}
          </p>
        </section>

        {/* Bot Client */}
        <section className="mt-6 rounded-xl border border-[var(--border)] bg-[var(--surface)]/80 p-6 ring-1 ring-[var(--accent)]/10">
          <div className="mb-3 flex items-center justify-between gap-3 rounded-lg border border-[var(--accent)]/20 bg-[var(--accent)]/10 px-3 py-2">
            <h2 className="text-sm font-medium text-[var(--accent)]">Live Practice (Bot)</h2>
            <div className="ml-auto rounded-md bg-[var(--accent)]/10 px-2 py-1 text-xs text-[var(--accent)]">
              Status: <span id="connection-status">{connectionStatus}</span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              id="connect-btn"
              className="rounded-md bg-[var(--accent)] px-3 py-2 text-sm font-medium text-white shadow-sm transition-colors hover:bg-[var(--accent-600)] active:bg-[var(--accent-700)] disabled:opacity-50"
              disabled={finishing}
            >
              Connect
            </button>
            {/* Disconnect button removed as it's not needed */}
            <button
              ref={finishBtnRef}
              onClick={async () => {
                if (finishing) return;
                if (!sessionId) {
                  alert("No active session. Please start and connect a session first.");
                  return;
                }

                setFinishing(true);
                setShowConfetti(true);

                // Prepare tile overlay config (origin near button) — match home animation
                const overlayMinMs = 700;
                const startTs = performance.now();
                const w = window.innerWidth;
                const h = window.innerHeight;
                const approxTile = 64;
                const cols = Math.max(8, Math.ceil(w / approxTile));
                const rows = Math.max(6, Math.ceil(h / approxTile));
                const rect = finishBtnRef.current?.getBoundingClientRect();
                const cx = rect ? rect.left + rect.width / 2 : w / 2;
                const cy = rect ? rect.top + rect.height / 2 : h / 2;
                const originX = Math.min(cols - 1, Math.max(0, Math.floor((cx / w) * cols)));
                const originY = Math.min(rows - 1, Math.max(0, Math.floor((cy / h) * rows)));
                const baseDelay = 18;
                const duration = 520;

                // Confetti runs briefly then tile overlay appears
                setTimeout(() => {
                  setGrid({ cols, rows, originX, originY, baseDelay, duration });
                  setAnimating(true);
                  setShowConfetti(false);
                  setAnimKey((k) => k + 1);
                }, 1000);

                // After a short pause for overlay animation, show end-of-convo modal
                const elapsed = performance.now() - startTs;
                if (elapsed < overlayMinMs) {
                  await new Promise((r) => setTimeout(r, overlayMinMs - elapsed));
                }
                setShowEndModal(true);
                setFinishing(false);
              }}
              className="rounded-md bg-[var(--accent)] px-3 py-2 text-sm font-medium text-white shadow-sm transition-colors hover:bg-[var(--accent-600)] active:bg-[var(--accent-700)] disabled:opacity-50"
              disabled={finishing || !sessionId}
            >
              Finish Conversation
            </button>
            <div className="ml-auto text-xs" style={{ color: "var(--text-muted)" }}>
              {webrtcReady ? "Ready" : "Loading client…"}
            </div>
          </div>

          {/* Media: Bot and Your Video side by side */}
          <div className="mt-4 grid gap-6 sm:grid-cols-2">
            <div>
              <label className="block text-xs font-medium">Bot Video</label>
              <video id="bot-video" autoPlay playsInline className="mt-2 h-64 w-full rounded-lg bg-black" />
              <audio id="bot-audio" autoPlay />
            </div>
            <div>
              <label className="block text-xs font-medium">Your Video</label>
              <div className="mt-2 relative">
                <video id="local-cam" autoPlay playsInline className="h-64 w-full rounded-lg bg-black" />
                <button id="mute-btn" className="absolute right-2 top-2 rounded-md bg-black/60 px-2 py-1 text-xs text-white">Toggle Cam</button>
              </div>
            </div>
          </div>

          {/* Device settings */}
          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <label className="block text-xs font-medium">Audio</label>
              <select id="audio-input" className="w-full rounded-md border border-[var(--border)] bg-[var(--muted)]/60 px-2 py-2 text-sm" defaultValue="">
                <option value="">Default device</option>
              </select>
              <select id="audio-codec" className="w-full rounded-md border border-[var(--border)] bg-[var(--muted)]/60 px-2 py-2 text-sm" defaultValue="default">
                <option value="default">Default codecs</option>
                <option value="opus/48000/2">Opus</option>
                <option value="PCMU/8000">PCMU</option>
                <option value="PCMA/8000">PCMA</option>
              </select>
              <button id="mute-mic" className="rounded-md border border-[var(--border)] px-3 py-2 text-sm">Unmute Mic</button>
            </div>
            <div className="space-y-2">
              <label className="block text-xs font-medium">Video</label>
              <select id="video-input" className="w-full rounded-md border border-[var(--border)] bg-[var(--muted)]/60 px-2 py-2 text-sm" defaultValue="">
                <option value="">Default device</option>
              </select>
              <select id="video-codec" className="w-full rounded-md border border-[var(--border)] bg-[var(--muted)]/60 px-2 py-2 text-sm" defaultValue="default">
                <option value="default">Default codecs</option>
                <option value="VP8/90000">VP8</option>
                <option value="H264/90000">H264</option>
              </select>
            </div>
          </div>

          {/* Local media section removed; your video shown above */}
        </section>

        <section className="mt-8 rounded-xl border border-[var(--border)] bg-[var(--surface)]/70 p-4">
          <div className="mb-3 flex items-center justify-between gap-3">
            <h2 className="text-sm font-medium">Stimulus</h2>
          </div>

          {!error && (
            <div className="mt-2">
              {/* Render trusted sanitized HTML from the server */}
              <div dangerouslySetInnerHTML={{ __html: html }} />
            </div>
          )}
          {error && (
            <div className="mt-3 text-sm text-red-500">
              {error}
            </div>
          )}
        </section>
      </div>

      {/* Confetti overlay */}
      {showConfetti && (
        <div aria-hidden className="pointer-events-none fixed inset-0 z-[9998] overflow-hidden">
          {Array.from({ length: 120 }).map((_, i) => {
            const left = Math.random() * 100;
            const delay = Math.round(Math.random() * 400);
            const dur = 1000 + Math.round(Math.random() * 600);
            const palette = ["#ff8a80", "#ffd180", "#ffff8d", "#ccff90", "#80d8ff", "#ea80fc"];
            const bg = palette[i % palette.length];
            const sx = Math.random() * 1.2 + 0.6;
            return (
              <div
                key={i}
                className="confetti-piece"
                style={{ left: `${left}%`, background: bg, transform: `scale(${sx})`, ["--delay" as any]: `${delay}ms`, ["--dur" as any]: `${dur}ms` }}
              />
            );
          })}
        </div>
      )}

      {/* Tile overlay (reuse home animation) */}
      {animating && grid && (
        <div
          aria-hidden
          key={animKey}
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

      {/* End-of-conversation modal */}
      {showEndModal && (
        <div className="fixed inset-0 z-[10000] flex items-center justify-center bg-black/40 p-4">
          <div className="w-full max-w-sm rounded-xl border border-[var(--border)] bg-[var(--surface)] p-5 shadow-xl">
            <h3 className="text-base font-medium">Have another conversation?</h3>
            <p className="mt-1 text-sm" style={{ color: "var(--text-muted)" }}>
              You can start a fresh practice right away.
            </p>
            <div className="mt-4 flex justify-end gap-2">
              <button
                onClick={() => {
                  // Dismiss modal and reveal the page again
                  setShowEndModal(false);
                  setAnimating(false);
                  setGrid(null);
                }}
                className="rounded-md border border-[var(--border)] px-3 py-2 text-sm"
              >
                Not now
              </button>
              <button
                onClick={async () => {
                  // Play a quick tile ripple, then navigate home
                  setShowEndModal(false);
                  const w = window.innerWidth;
                  const h = window.innerHeight;
                  const approxTile = 64;
                  const cols = Math.max(8, Math.ceil(w / approxTile));
                  const rows = Math.max(6, Math.ceil(h / approxTile));
                  const originX = Math.floor(cols / 2);
                  const originY = Math.floor(rows / 2);
                  const baseDelay = 14;
                  const duration = 420;
                  setGrid({ cols, rows, originX, originY, baseDelay, duration });
                  setAnimating(true);
                  setAnimKey((k) => k + 1);
                  await new Promise((r) => setTimeout(r, duration + 250));
                  router.push("/");
                }}
                className="rounded-md bg-[var(--accent)] px-3 py-2 text-sm font-medium text-white shadow-sm hover:bg-[var(--accent-600)] active:bg-[var(--accent-700)]"
              >
                Yes
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
