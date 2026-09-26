/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        lab: {
          950: '#09090b', // Deep zinc/black (ChatGPT background)
          900: '#111113', // Neutral dark panel
          850: '#18181b', // Surface / Card background
          800: '#222226', // Elevated background / Hover
          750: '#2a2a30', // Active item
          700: '#333338', // Hairline border
          600: '#484852', // Border highlight
          500: '#71717a', // Muted secondary text
          400: '#a1a1aa', // Subtle gray
          300: '#d4d4d8', // Light gray text
          200: '#e4e4e7', // Near white
          100: '#f4f4f5', // Crisp light
          accent: '#ffffff', // Pure crisp white accent
          accentHover: '#e4e4e7',
          accent2: '#10a37f', // Subtle ChatGPT signature teal/green
          ok: '#10a37f',
          warn: '#f59e0b',
          err: '#ef4444',
        },
      },
      fontFamily: {
        sans: ['Inter', 'Outfit', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      boxShadow: {
        'subtle': '0 1px 3px 0 rgba(0, 0, 0, 0.4), 0 1px 2px -1px rgba(0, 0, 0, 0.3)',
        'elevated': '0 8px 30px rgba(0, 0, 0, 0.6)',
        'glow-white': '0 0 15px rgba(255, 255, 255, 0.12)',
      },
    },
  },
  plugins: [],
}
