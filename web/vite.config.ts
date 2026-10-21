import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

const runnerTarget = process.env.VLLM_EXPLORER_RUNNER_URL || "http://127.0.0.1:8765"

export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    fs: { allow: [".."] },
    proxy: {
      "/api": {
        target: runnerTarget,
        timeout: 2_100_000,
        proxyTimeout: 2_100_000,
      },
    },
  },
  preview: {
    host: "127.0.0.1",
    port: 4173,
    proxy: {
      "/api": {
        target: runnerTarget,
        timeout: 2_100_000,
        proxyTimeout: 2_100_000,
      },
    },
  },
})
