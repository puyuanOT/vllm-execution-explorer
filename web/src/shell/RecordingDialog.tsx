import type { CatalogEntry } from "../api.ts"

export function RecordingDialog({
  catalog,
  catalogError,
  activeId,
  onOpen,
  onFiles,
}: {
  catalog: CatalogEntry[]
  catalogError: string
  activeId: string
  onOpen: (id: string) => void
  onFiles: (files: FileList) => void
}) {
  return (
    <div className="dialog-body">
      <h2 id="recordings-title">Recordings</h2>
      <p className="lede">Bundled lessons and saved runs. Opening one does not change the arrival draft.</p>
      {catalogError ? <p className="field-error" role="alert">{catalogError}</p> : null}
      <ul className="lesson-list">
        {catalog.map((item) => (
          <li key={item.id}>
            <button type="button" aria-pressed={activeId === item.id} onClick={() => onOpen(item.id)}>
              <strong>{item.title}</strong>
              <span>{item.summary}</span>
            </button>
          </li>
        ))}
      </ul>
      <label className="file">
        Open manifest.json, events.jsonl, and optional content.jsonl
        <input
          type="file"
          multiple
          accept=".json,.jsonl,application/json"
          onChange={(event) => {
            if (event.target.files && event.target.files.length > 0) onFiles(event.target.files)
          }}
        />
      </label>
    </div>
  )
}
