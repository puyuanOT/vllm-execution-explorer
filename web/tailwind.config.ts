import type { Config } from "tailwindcss"

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas: "#06080c",
        surface: "#11151d",
        raised: "#181e29",
        line: "#29313f",
        ink: "#e8edf5",
        muted: "#a2afc2",
        accent: "#77b7ff",
      },
    },
  },
  plugins: [],
} satisfies Config
