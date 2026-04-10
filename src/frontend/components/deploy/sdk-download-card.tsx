import type { SdkDownloadInfo } from "@/lib/types"

interface SdkDownloadCardProps {
  info: SdkDownloadInfo
}

const PYTHON_USAGE = (className: string) => `from ${className.toLowerCase()}_sdk import ${className}

predictor = ${className}(base_url="http://localhost:8000")
result = predictor.predict(feature1=1.0, feature2="value")
print(result["prediction"])`

const JS_USAGE = (className: string) => `import { ${className} } from './${className.toLowerCase()}_sdk.js';

const predictor = new ${className}({ baseUrl: 'http://localhost:8000' });
const result = await predictor.predict({ feature1: 1.0, feature2: 'value' });
console.log(result.prediction);`

export function SdkDownloadCard({ info }: SdkDownloadCardProps) {
  const algorithmLabel = info.algorithm.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase())
  const problemLabel = info.problem_type === "classification" ? "Classification" : "Regression"

  return (
    <div
      role="region"
      className="mt-2 rounded-lg border border-indigo-200 bg-indigo-50/50 p-3 text-sm"
      aria-label="SDK download card"
    >
      <div className="mb-2 flex items-center gap-2">
        <span aria-hidden="true">📦</span>
        <span className="font-semibold text-indigo-900">Developer SDK</span>
        <span className="rounded bg-indigo-100 px-2 py-0.5 text-xs font-medium text-indigo-700">
          {problemLabel}
        </span>
        <span className="rounded bg-slate-100 px-2 py-0.5 text-xs text-slate-600">
          {algorithmLabel}
        </span>
      </div>

      <p className="mb-3 text-xs text-indigo-700">
        Share these files with your developer so they can call the{" "}
        <code className="rounded bg-indigo-100 px-1 font-mono">{info.target_column}</code> model
        without writing HTTP code from scratch.
      </p>

      {/* Download buttons */}
      <div className="mb-3 flex flex-wrap gap-2">
        <a
          href={info.python_url}
          download
          className="inline-flex items-center gap-1.5 rounded border border-indigo-300 bg-white px-3 py-1.5 text-xs font-medium text-indigo-700 transition-colors hover:bg-indigo-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
          aria-label="Download Python SDK"
        >
          <span aria-hidden="true">⬇</span>
          Python SDK (.py)
        </a>
        <a
          href={info.javascript_url}
          download
          className="inline-flex items-center gap-1.5 rounded border border-slate-300 bg-white px-3 py-1.5 text-xs font-medium text-slate-700 transition-colors hover:bg-slate-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-500"
          aria-label="Download JavaScript SDK"
        >
          <span aria-hidden="true">⬇</span>
          JavaScript SDK (.js)
        </a>
      </div>

      {/* Usage preview */}
      <div className="space-y-2">
        <p className="text-xs font-medium text-indigo-800">Python usage:</p>
        <pre className="overflow-x-auto rounded bg-slate-900 p-2 text-xs leading-relaxed text-slate-100">
          <code>{PYTHON_USAGE(info.class_name)}</code>
        </pre>
        <p className="text-xs font-medium text-indigo-800">JavaScript usage:</p>
        <pre className="overflow-x-auto rounded bg-slate-900 p-2 text-xs leading-relaxed text-slate-100">
          <code>{JS_USAGE(info.class_name)}</code>
        </pre>
      </div>

      <p className="mt-2 text-xs text-slate-500">
        Class name: <code className="font-mono">{info.class_name}</code> — both SDKs include{" "}
        <code className="font-mono">predict()</code> and{" "}
        <code className="font-mono">predictBatch()</code> methods with typed parameters.
      </p>
    </div>
  )
}
