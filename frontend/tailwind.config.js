/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        lab: {
          950: '#06040c', // Deep void black
          900: '#0c0818', // Obsidian violet surface
          850: '#130d24', // Card & panel surface
          800: '#1c1334', // Elevated surface
          750: '#261845', // Highlight surface / active
          700: '#382264', // Neon-tinted border
          600: '#523193', // Active border / glow ring
          500: '#7545ce', // Muted violet
          accent: '#ec4899', // Cyber neon pink
          accentHover: '#f43f5e', // Hot pink glow
          accent2: '#a855f7', // Electric purple
          accent2Hover: '#c084fc', // Luminous lavender
          pink: '#ec4899',
          pinkLight: '#f472b6',
          purple: '#a855f7',
          purpleLight: '#c084fc',
          cyan: '#22d3ee',
          ok: '#10b981',
          warn: '#f59e0b',
          err: '#f43f5e',
        },
      },
      fontFamily: {
        sans: ['Outfit', 'system-ui', '-apple-system', 'BlinkMacSystemFont', 'sans-serif'],
        mono: ['JetBrains Mono', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
      },
      boxShadow: {
        'neon-pink': '0 0 15px rgba(236, 72, 153, 0.45)',
        'neon-purple': '0 0 15px rgba(168, 85, 247, 0.45)',
        'neon-cyan': '0 0 15px rgba(34, 211, 238, 0.45)',
        'glass': '0 8px 32px 0 rgba(0, 0, 0, 0.45)',
      },
    },
  },
  plugins: [],
}
