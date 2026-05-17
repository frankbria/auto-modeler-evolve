import type { RollbackChatResult } from "@/lib/types"

interface RollbackChatCardProps {
  result: RollbackChatResult
}

function AlgoName(raw: string | null): string {
  if (!raw) return "Unknown"
  return raw.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
}

export function RollbackChatCard({ result }: RollbackChatCardProps) {
  const {
    rolled_back,
    rolled_back_to_version,
    current_version_number,
    error_message,
    versions,
    total_versions,
  } = result

  const borderColor = rolled_back
    ? "border-emerald-200 bg-emerald-50/50"
    : error_message
      ? "border-rose-200 bg-rose-50/50"
      : "border-indigo-200 bg-indigo-50/50"

  const headerColor = rolled_back
    ? "text-emerald-900"
    : error_message
      ? "text-rose-900"
      : "text-indigo-900"

  return (
    <div
      role="region"
      className={`mt-2 rounded-lg border ${borderColor} p-3 text-sm`}
      aria-label="Deployment version card"
    >
      <div className="mb-2 flex items-center gap-2">
        <span aria-hidden="true">🔄</span>
        <span className={`font-semibold ${headerColor}`}>
          {rolled_back ? "Rollback Complete" : "Deployment Versions"}
        </span>
        <span className="rounded bg-indigo-100 px-2 py-0.5 text-xs font-medium text-indigo-700">
          {total_versions} version{total_versions !== 1 ? "s" : ""}
        </span>
        {rolled_back && (
          <span className="rounded bg-emerald-100 px-2 py-0.5 text-xs font-medium text-emerald-700">
            ✓ Rolled back to v{rolled_back_to_version}
          </span>
        )}
      </div>

      {error_message && (
        <div
          role="alert"
          className="mb-3 rounded bg-rose-100/60 px-3 py-2 text-xs text-rose-800"
        >
          <span aria-hidden="true">⚠️</span> {error_message}
        </div>
      )}

      {rolled_back && (
        <div className="mb-3 rounded bg-emerald-100/60 px-3 py-2 text-xs text-emerald-900">
          <strong>Endpoint URL unchanged.</strong> Your deployment is now serving
          version {rolled_back_to_version}. All existing integrations continue
          to work.
        </div>
      )}

      {versions.length > 0 && (
        <div className="space-y-1">
          <p className="mb-1 text-xs font-medium text-slate-600">
            Version History
          </p>
          {versions.map((v) => (
            <div
              key={v.version_number}
              className={`flex items-center justify-between rounded px-2 py-1.5 text-xs ${
                v.is_current
                  ? "bg-indigo-100/70 font-medium"
                  : "bg-slate-50"
              }`}
              data-testid={`version-row-${v.version_number}`}
            >
              <div className="flex items-center gap-2">
                <span className="font-mono text-slate-700">v{v.version_number}</span>
                {v.algorithm && (
                  <span className="text-slate-600">{AlgoName(v.algorithm)}</span>
                )}
                {v.is_current && (
                  <span className="rounded bg-indigo-200 px-1.5 py-0.5 text-xs text-indigo-800">
                    Current
                  </span>
                )}
                {v.version_number === rolled_back_to_version && rolled_back && (
                  <span className="rounded bg-emerald-200 px-1.5 py-0.5 text-xs text-emerald-800">
                    ✓ Restored
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2 text-slate-500">
                {v.metric_display && (
                  <span className="font-mono text-xs">{v.metric_display}</span>
                )}
                {v.deployed_at && (
                  <span>
                    {new Date(v.deployed_at).toLocaleDateString(undefined, {
                      month: "short",
                      day: "numeric",
                    })}
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {!rolled_back && total_versions > 1 && (
        <p className="mt-3 border-t border-indigo-100 pt-2 text-xs text-slate-500">
          To roll back, say &ldquo;roll back to version{" "}
          {versions.find((v) => !v.is_current)?.version_number ?? "N"}
          &rdquo;. Your prediction endpoint URL will stay the same.
        </p>
      )}

      {!rolled_back && total_versions <= 1 && !error_message && (
        <p className="mt-2 text-xs text-slate-500">
          No previous versions to roll back to. Retrain and redeploy to create a
          new version.
        </p>
      )}

      {rolled_back && (
        <p className="mt-3 border-t border-emerald-100 pt-2 text-xs text-slate-500">
          Current version is now v{current_version_number}. The previous version
          remains in history — you can roll back to it again at any time.
        </p>
      )}
    </div>
  )
}
