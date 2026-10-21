export function focusValidationField(fieldId: string) {
  window.requestAnimationFrame(() => {
    const field = document.getElementById(fieldId)
    if (!field) return
    for (let parent = field.parentElement; parent; parent = parent.parentElement) {
      if (parent instanceof HTMLDetailsElement) parent.open = true
    }
    field.focus()
  })
}
