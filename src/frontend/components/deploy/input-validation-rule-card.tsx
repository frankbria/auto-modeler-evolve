import type { InputValidationRuleEntry, InputValidationRuleResult } from "@/lib/types"

interface InputValidationRuleCardProps {
  result: InputValidationRuleResult
}

function RuleTypeBadge({ ruleType }: { ruleType: string }) {
  const labels: Record<string, string> = {
    range: "Range",
    one_of: "One of",
    not_null: "Required",
  }
  return (
    <span className="rounded bg-violet-100 px-2 py-0.5 text-xs font-medium text-violet-700">
      {labels[ruleType] ?? ruleType}
    </span>
  )
}

function RuleRow({ rule }: { rule: InputValidationRuleEntry }) {
  return (
    <div
      className="flex items-start justify-between gap-2 rounded border border-slate-100 bg-slate-50 px-3 py-2 text-xs"
      data-testid={`validation-rule-row-${rule.feature_name}`}
    >
      <div className="flex min-w-0 flex-col gap-0.5">
        <code className="font-semibold text-slate-800">{rule.feature_name}</code>
        <span className="text-slate-600">{rule.description}</span>
      </div>
      <RuleTypeBadge ruleType={rule.rule_type} />
    </div>
  )
}

export function InputValidationRuleCard({ result }: InputValidationRuleCardProps) {
  const { action, summary } = result

  const borderColor =
    action === "created"
      ? "border-violet-200 bg-violet-50/50"
      : action === "deleted"
        ? "border-rose-200 bg-rose-50/50"
        : "border-slate-200 bg-slate-50/50"

  const iconAndTitle =
    action === "created"
      ? { icon: "🛡️", title: "Validation Rule Added" }
      : action === "deleted"
        ? { icon: "🗑️", title: "Validation Rules Removed" }
        : action === "list"
          ? { icon: "📋", title: "Input Validation Rules" }
          : { icon: "💡", title: "Validation Rule Guidance" }

  return (
    <div
      role="region"
      className={`mt-2 rounded-lg border p-3 text-sm ${borderColor}`}
      aria-label="Input validation rule card"
    >
      {/* Header */}
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <span aria-hidden="true">{iconAndTitle.icon}</span>
        <span className="font-semibold text-slate-900">{iconAndTitle.title}</span>
        {action === "created" && result.total_rules !== undefined && (
          <span className="rounded bg-violet-100 px-2 py-0.5 text-xs text-violet-700">
            {result.total_rules} rule{result.total_rules !== 1 ? "s" : ""} active
          </span>
        )}
        {action === "list" && result.count !== undefined && (
          <span className="rounded bg-slate-100 px-2 py-0.5 text-xs text-slate-600">
            {result.count} rule{result.count !== 1 ? "s" : ""}
          </span>
        )}
        {action === "deleted" && result.deleted_count !== undefined && (
          <span className="rounded bg-rose-100 px-2 py-0.5 text-xs text-rose-700">
            {result.deleted_count} removed
          </span>
        )}
      </div>

      {/* Summary */}
      <p className="mb-3 text-xs text-slate-700">{summary}</p>

      {/* Created rule detail */}
      {action === "created" && result.feature_name && result.rule_type && (
        <div className="mb-3 rounded border border-violet-100 bg-white px-3 py-2 text-xs">
          <div className="flex items-center gap-2">
            <code className="font-semibold text-slate-800">{result.feature_name}</code>
            <RuleTypeBadge ruleType={result.rule_type} />
          </div>
          {result.description && (
            <p className="mt-1 text-slate-600">{result.description}</p>
          )}
        </div>
      )}

      {/* List of rules */}
      {action === "list" && result.rules && result.rules.length > 0 && (
        <div className="mb-3 space-y-1.5">
          {result.rules.map((rule) => (
            <RuleRow key={rule.id} rule={rule} />
          ))}
        </div>
      )}

      {action === "list" && (!result.rules || result.rules.length === 0) && (
        <p className="mb-3 text-xs text-slate-400 italic">No validation rules configured.</p>
      )}

      {/* Footer hint */}
      <p className="mt-2 border-t border-slate-100 pt-2 text-xs text-slate-400">
        {action === "list" || action === "guidance"
          ? "Try: \"validate that age is between 0 and 120\", \"require region to be one of East, West\", or \"remove the validation rules\"."
          : "The prediction API now rejects inputs that violate this rule with a plain-English 422 error."}
      </p>
    </div>
  )
}
