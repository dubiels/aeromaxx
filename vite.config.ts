import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { readFileSync } from 'fs'

export default defineConfig({
  plugins: [
    react(),
    {
      name: 'mediapipe-sourcemap-fix',
      enforce: 'pre',
      load(id) {
        const file = id.split('?')[0]
        if (!file.includes('@mediapipe')) return null
        if (!file.endsWith('.js') && !file.endsWith('.mjs') && !file.endsWith('.cjs')) return null
        try {
          const code = readFileSync(file, 'utf-8')
          return {
            code: code.replace(/\/\/# sourceMappingURL=\S+/g, ''),
            map: null,
          }
        } catch {
          return null
        }
      },
    },
  ],
  optimizeDeps: {
    exclude: ['@mediapipe/tasks-vision'],
  },
  server: {
    proxy: {
      '/meshy-assets': {
        target: 'https://assets.meshy.ai',
        changeOrigin: true,
        rewrite: path => path.replace(/^\/meshy-assets/, ''),
      },
    },
  },
})