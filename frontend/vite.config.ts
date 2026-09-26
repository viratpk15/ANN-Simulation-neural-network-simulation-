import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import type { ConfigEnv, ProxyOptions, UserConfig } from 'vite'

// Ports probed for the backend when BACKEND_PORT is not set.
const CANDIDATE_PORTS = ['8000', '8001', '8002', '8003']

// True only when this port answers as the NeuroSim Lab API.
async function isNeurosim(port: string): Promise<boolean> {
  try {
    const res = await fetch('http://127.0.0.1:' + port + '/api/health', {
      signal: AbortSignal.timeout(600),
    })
    if (!res.ok) return false
    const body = (await res.json()) as { name?: string }
    return body?.name === 'NeuroSim Lab API'
  } catch {
    return false
  }
}

async function detectBackendPort(): Promise<string> {
  if (process.env.BACKEND_PORT) return process.env.BACKEND_PORT
  for (const port of CANDIDATE_PORTS) {
    if (await isNeurosim(port)) return port
  }
  return '8000'
}

// Dev proxy for /api and /ws (production serves the SPA from FastAPI itself).
// http-proxy re-reads target from the options object on every request, so the
// watchdog can re-point the proxy at runtime when the backend port changes or
// was not listening yet when vite started.
export default defineConfig(async ({ command }: ConfigEnv): Promise<UserConfig> => {
  let backendPort = await detectBackendPort()

  const apiProxy: ProxyOptions = {
    target: 'http://localhost:' + backendPort,
    changeOrigin: true,
  }
  const wsProxy: ProxyOptions = {
    target: 'ws://localhost:' + backendPort,
    changeOrigin: true,
    ws: true,
  }

  if (command === 'serve' && !process.env.BACKEND_PORT) {
    const timer = setInterval(() => {
      void (async () => {
        if (await isNeurosim(backendPort)) return
        for (const port of CANDIDATE_PORTS) {
          if (port === backendPort) continue
          if (await isNeurosim(port)) {
            backendPort = port
            apiProxy.target = 'http://localhost:' + port
            wsProxy.target = 'ws://localhost:' + port
            console.log('[vite] backend re-detected on :' + port)
            return
          }
        }
      })()
    }, 5000)
    const t = timer as unknown as { unref?: () => void }
    if (typeof t.unref === 'function') t.unref()
  }

  return {
    plugins: [react()],
    server: {
      host: true,
      port: 5173,
      allowedHosts: true,
      proxy: { '/api': apiProxy, '/ws': wsProxy },
    },
    preview: { host: true, port: 4173 },
    build: { chunkSizeWarningLimit: 1500 },
  }
})
