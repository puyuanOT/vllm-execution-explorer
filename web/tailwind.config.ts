import type { Config } from "tailwindcss"

const steel = {
  50: "#f3f6fa",
  100: "#e2eaf4",
  200: "#c9d4e2",
  300: "#aebdd0",
  400: "#95a5bb",
  500: "#63738c",
  600: "#3e4d65",
  700: "#1c2a41",
  800: "#131e30",
  900: "#0c1320",
  950: "#070b13",
}

const signal = {
  DEFAULT: "#3fcde6",
  50: "#e9fbfe",
  100: "#c8f3fa",
  200: "#9fe8f5",
  300: "#6fdbef",
  400: "#3fcde6",
  500: "#2bb6cf",
  600: "#2398ad",
  700: "#1d7a8b",
  800: "#175d6a",
  900: "#103f48",
  950: "#082a31",
}

const green = {
  100: "#d9f5e8",
  200: "#b0e9cf",
  300: "#86dcb6",
  400: "#5fcf9f",
  500: "#46b386",
  600: "#368f6b",
  700: "#2a6e53",
  800: "#1f4f3c",
  900: "#163a2d",
  950: "#0b2119",
}

const amber = {
  100: "#f6ead1",
  200: "#efd7a6",
  300: "#e7c27b",
  400: "#dfad57",
  500: "#c99540",
  600: "#a47833",
  700: "#7d5b28",
  800: "#5a421e",
  900: "#3d2d15",
  950: "#24190b",
}

const red = {
  100: "#f8dbe0",
  200: "#f1b7c1",
  300: "#ea8fa0",
  400: "#e5687d",
  500: "#cf5167",
  600: "#a93f52",
  700: "#82313f",
  800: "#5d232d",
  900: "#3f1820",
  950: "#250d12",
}

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    borderRadius: {
      none: "0",
      sm: "1px",
      DEFAULT: "2px",
      md: "2px",
      lg: "3px",
      xl: "3px",
      "2xl": "4px",
      full: "9999px",
    },
    extend: {
      colors: {
        slate: steel,
        blue: signal,
        sky: signal,
        accent: signal,
        emerald: green,
        amber,
        red,
        rose: red,
        canvas: "#04070c",
        surface: "#080d16",
        raised: "#0c1320",
        line: "#131e30",
        "line-strong": "#1c2a41",
        ink: "#e2eaf4",
        muted: "#95a5bb",
        faint: "#63738c",
        explain: "#9d90ec",
        synthetic: "#d46cc0",
      },
      fontFamily: {
        sans: ["Space Grotesk", "Segoe UI", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
      },
    },
  },
  plugins: [],
} satisfies Config
