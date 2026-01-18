/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        biotuner: {
          primary: '#00d9ff',
          secondary: '#7c3aed',
          accent: '#10b981',
          dark: {
            900: '#0a0a0a',
            800: '#121212',
            700: '#1a1a1a',
            600: '#252525',
          },
          light: '#e5e7eb',
        }
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'gradient-conic': 'conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))',
      },
    },
  },
  plugins: [],
}
