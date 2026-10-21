import { AlertDialog } from "@base-ui/react/alert-dialog"

export function ConfirmDialog({
  open,
  onOpenChange,
  title,
  description,
  confirmLabel,
  onConfirm,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: string
  description: string
  confirmLabel: string
  onConfirm: () => void
}) {
  return (
    <AlertDialog.Root open={open} onOpenChange={onOpenChange}>
      <AlertDialog.Portal>
        <AlertDialog.Backdrop className="fixed inset-0 z-40 bg-black/70" />
        <AlertDialog.Viewport className="safe-dialog-viewport fixed inset-0 z-50 flex items-center justify-center overflow-y-auto p-4">
          <AlertDialog.Popup className="w-full max-w-md rounded-xl border border-slate-700 bg-slate-950 p-5 text-slate-100 shadow-xl outline-none">
            <AlertDialog.Title className="text-balance text-lg font-semibold">{title}</AlertDialog.Title>
            <AlertDialog.Description className="mt-2 text-pretty text-sm text-slate-400">{description}</AlertDialog.Description>
            <div className="mt-5 flex justify-end gap-2">
              <AlertDialog.Close className="rounded-md border border-slate-700 bg-slate-900 px-4 py-2 text-sm text-slate-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-400">
                Keep draft
              </AlertDialog.Close>
              <button
                type="button"
                className="rounded-md border border-rose-800 bg-rose-950 px-4 py-2 text-sm font-medium text-rose-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-rose-400"
                onClick={() => {
                  onConfirm()
                  onOpenChange(false)
                }}
              >
                {confirmLabel}
              </button>
            </div>
          </AlertDialog.Popup>
        </AlertDialog.Viewport>
      </AlertDialog.Portal>
    </AlertDialog.Root>
  )
}
