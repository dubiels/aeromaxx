import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { readFileSync } from 'fs'

export default defineConfig({
  plugins: [
    react(),
    {
      // @mediapipe/tasks-vision ships without its .map files.
      // enforce:'pre' ensures this load hook runs before Vite's own file reader,
      // so extractSourcemapFromFile never sees the sourceMappingURL comment.
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
})
