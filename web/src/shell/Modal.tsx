import { Dialog } from "@base-ui/react/dialog"
import { X } from "lucide-react"
import type { ReactNode } from "react"

import { cn } from "../lib/cn.ts"

export function Modal({
  open,
  onOpenChange,
  title,
  description,
  children,
  placement = "center",
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: string
  description: string
  children: ReactNode
  placement?: "center" | "right"
}) {
  return (
    <Dialog.Root open={open} onOpenChange={onOpenChange}>
      <Dialog.Portal>
        <Dialog.Backdrop className="fixed inset-0 z-40 bg-black/70" />
        <Dialog.Viewport
          className={cn(
            "safe-dialog-viewport fixed inset-0 z-50 flex overflow-y-auto p-4",
            placement === "right" ? "items-stretch justify-end" : "items-center justify-center",
          )}
        >
          <Dialog.Popup
            className={cn(
              "relative max-h-[calc(100dvh-2rem)] overflow-y-auto rounded-xl border border-slate-700 bg-slate-950 p-5 text-slate-100 shadow-xl outline-none",
              placement === "right" ? "h-full w-full max-w-md" : "w-full max-w-xl",
            )}
          >
            <div className="mb-4 pr-10">
              <Dialog.Title className="text-balance text-lg font-semibold">{title}</Dialog.Title>
              <Dialog.Description className="mt-1 text-pretty text-sm text-slate-400">
                {description}
              </Dialog.Description>
            </div>
            <Dialog.Close
              className="absolute right-4 top-4 grid size-8 place-items-center rounded-md border border-slate-700 bg-slate-900 text-slate-300 hover:text-white focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-400"
              aria-label={`Close ${title}`}
            >
              <X aria-hidden="true" className="size-4" />
            </Dialog.Close>
            {children}
          </Dialog.Popup>
        </Dialog.Viewport>
      </Dialog.Portal>
    </Dialog.Root>
  )
}
