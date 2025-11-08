This is a [Next.js](https://nextjs.org) project bootstrapped with [`create-next-app`](https://nextjs.org/docs/app/api-reference/cli/create-next-app).

## Getting Started

First, run the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
# or
bun dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

This project uses [`next/font`](https://nextjs.org/docs/app/building-your-application/optimizing/fonts) to automatically optimize and load [Geist](https://vercel.com/font), a new font family for Vercel.

## Bot Integration (/learn)

The `/learn` page includes a Pipecat WebRTC client UI to connect to a local bot server from `../../pipecat-examples/p2p-webrtc/video-transform`.

Prerequisites:
- Start the Python bot server in `../../pipecat-examples/p2p-webrtc/video-transform/server`:
  - Create/activate a virtualenv
  - `pip install -r requirements.txt`
  - `python server.py --host localhost --port 7860`
- Install client libraries in this Next.js app:
  - `npm install @pipecat-ai/client-js @pipecat-ai/small-webrtc-transport`

Config:
- Add to `.env.local` if needed (defaults to localhost:7860):

```
PIPECAT_SERVER_ORIGIN=http://localhost:7860
```

How it works:
- Next.js API route `/api/offer` proxies POST/PATCH to `${PIPECAT_SERVER_ORIGIN}/api/offer` for WebRTC signaling.
- The `/learn` page dynamically loads the Pipecat client and provides connect/disconnect, device selection, and local/bot media views.

Notes:
- If the client libraries are not installed, the WebRTC client will not initialize but the page will still render.
- Ensure the browser is allowed to use microphone and camera.

## OpenRouter Streaming Stimulus

This app streams a text-based stimulus on the `/learn` page based on the scenario you enter on the home page and the "Start Practice" flow.

1. Create a `.env.local` in the project root and add your key:

```
OPENROUTER_API_KEY=sk-or-...
# Optional:
# OPENROUTER_REFERER=https://your-site.example
# OPENROUTER_TITLE=Your Site Name
# OPENROUTER_MODEL=openai/gpt-4o-mini
```

2. Run `npm run dev` and open the app.

When you submit the scenario, the app navigates to `/learn` and starts streaming the LLM output in real time.

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

You can check out [the Next.js GitHub repository](https://github.com/vercel/next.js) - your feedback and contributions are welcome!

## Deploy on Vercel

The easiest way to deploy your Next.js app is to use the [Vercel Platform](https://vercel.com/new?utm_medium=default-template&filter=next.js&utm_source=create-next-app&utm_campaign=create-next-app-readme) from the creators of Next.js.

Check out our [Next.js deployment documentation](https://nextjs.org/docs/app/building-your-application/deploying) for more details.
