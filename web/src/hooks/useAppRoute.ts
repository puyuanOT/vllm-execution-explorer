import { useCallback, useEffect, useState } from "react"

export type AppRoute = "configure" | "explore"

function routeFromLocation(): AppRoute {
  return window.location.pathname === "/explore" ? "explore" : "configure"
}

export function useAppRoute() {
  const [route, setRoute] = useState<AppRoute>(routeFromLocation)

  useEffect(() => {
    const onPopState = () => setRoute(routeFromLocation())
    window.addEventListener("popstate", onPopState)
    return () => window.removeEventListener("popstate", onPopState)
  }, [])

  useEffect(() => {
    document.title = `${route === "explore" ? "Explore" : "Configure"} · vLLM Execution Explorer`
  }, [route])

  const navigate = useCallback((next: AppRoute, options?: { replace?: boolean }) => {
    const path = next === "explore" ? "/explore" : "/"
    if (options?.replace) window.history.replaceState(null, "", path)
    else if (window.location.pathname !== path) window.history.pushState(null, "", path)
    setRoute(next)
    window.scrollTo({ top: 0 })
  }, [])

  return { route, navigate }
}
