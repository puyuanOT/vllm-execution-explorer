import { useEffect, useState } from "react"

import { emptyScenario, staggeredExample } from "../scenario/example.ts"
import { reduceScenario, type ScenarioAction } from "../scenario/reducer.ts"
import { clearScenarioStorage, loadScenario, saveScenario } from "../scenario/storage.ts"
import { createId } from "../scenario/ids.ts"
import type { ScenarioDraft } from "../scenario/types.ts"

export function useScenarioDraft() {
  const [boot] = useState(() => loadScenario())
  const [scenario, setScenario] = useState<ScenarioDraft>(boot.value)
  const [undo, setUndo] = useState<ScenarioDraft[]>([])
  const [storageNote, setStorageNote] = useState(boot.note)
  const [storageError, setStorageError] = useState(boot.error)

  useEffect(() => {
    const saved = saveScenario(scenario)
    setStorageNote(saved.note)
    setStorageError(saved.error)
  }, [scenario])

  function dispatch(action: ScenarioAction) {
    setScenario((current) => {
      if (action.type === "delete-batch" || action.type === "delete-sample") {
        setUndo((stack) => [...stack, current].slice(-20))
      }
      return reduceScenario(current, action)
    })
  }

  function undoDelete() {
    setUndo((stack) => {
      const previous = stack[stack.length - 1]
      if (previous) setScenario(previous)
      return stack.slice(0, -1)
    })
  }

  function startEmpty() {
    setUndo((stack) => [...stack, scenario].slice(-20))
    const next = emptyScenario()
    next.id = createId("scenario")
    setScenario(next)
  }

  function clearDraft() {
    const cleared = clearScenarioStorage()
    setStorageError(cleared.error)
    setStorageNote(cleared.note || "Draft cleared on this device")
    const next = staggeredExample()
    next.id = createId("scenario")
    setScenario(next)
    setUndo([])
  }

  function replace(next: ScenarioDraft) {
    setScenario(next)
  }

  return {
    scenario,
    dispatch,
    undoDelete,
    canUndo: undo.length > 0,
    startEmpty,
    clearDraft,
    replace,
    storageNote,
    storageError,
  }
}
