import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

/**
 * Конфигурация для сборки встраиваемого виджета.
 * Результат — один JS-файл, который можно подключить через <script> тег.
 */
export default defineConfig({
  plugins: [react()],
  define: {
    'process.env.NODE_ENV': '"production"',
  },
  build: {
    lib: {
      entry: 'src/widget/embed.tsx',
      name: 'FarmbazisSupport',
      fileName: 'farmbazis-support-widget',
      formats: ['iife'],
    },
    outDir: 'dist-widget',
    cssCodeSplit: false,
    rollupOptions: {
      output: {
        inlineDynamicImports: true,
      },
    },
  },
})
