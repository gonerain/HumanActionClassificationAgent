import { defineConfig } from 'vite'

// Dev proxy to backend on localhost:8000
export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      // Proxy API + WS only. Do NOT proxy '/'
      '/cameras': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        ws: true,
      },
      '/dwell_events': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})
