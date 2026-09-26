/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        lab: {
          950: '#0b0f1a',
          900: '#0f1524',
          850: '#131a2e',
          800: '#182136',
          700: '#233052',
          600: '#2f3f69',
          accent: '#38bdf8',
          accent2: '#a78bfa',
          ok: '#34d399',
          warn: '#fbbf24',
          err: '#f87171',
        },
      },
      fontFamily: {
        mono: ['ui-monospace', 'SFMono-Regular', 'Menlo', 'Consolas', 'monospace'],
      },
    },
  },
  plugins: [],
}
