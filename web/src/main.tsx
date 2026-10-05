import { createRoot } from "react-dom/client"

import "@fontsource/space-grotesk/400.css"
import "@fontsource/space-grotesk/500.css"
import "@fontsource/space-grotesk/600.css"
import "@fontsource/jetbrains-mono/400.css"
import "@fontsource/jetbrains-mono/500.css"
import "@fontsource/jetbrains-mono/600.css"
import { App } from "./App.tsx"
import "./styles.css"

const root = document.getElementById("root")
if (!root) throw new Error("missing root")
createRoot(root).render(<App />)
