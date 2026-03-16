"use client"

/**
 * WorkflowProgress — horizontal stepper showing where the user is in the
 * Upload → Train → Validate → Deploy workflow.
 *
 * Each step is clickable (calls onStepClick) so the user can jump directly
 * to the relevant right-panel tab. The active step is the first incomplete one.
 */

interface Step {
  key: string
  label: string
  description: string
  tab: string
}

const STEPS: Step[] = [
  { key: "upload", label: "Upload", description: "Load your data", tab: "data" },
  { key: "train", label: "Train", description: "Build a model", tab: "models" },
  { key: "validate", label: "Validate", description: "Review accuracy", tab: "validate" },
  { key: "deploy", label: "Deploy", description: "Share predictions", tab: "deploy" },
]

interface WorkflowProgressProps {
  hasDataset: boolean
  hasSelectedModel: boolean
  hasDeployment: boolean
  /** Called when the user clicks a step pill */
  onStepClick?: (tab: string) => void
}

function stepStatus(
  stepKey: string,
  hasDataset: boolean,
  hasSelectedModel: boolean,
  hasDeployment: boolean
): "done" | "active" | "pending" {
  switch (stepKey) {
    case "upload":
      return hasDataset ? "done" : "active"
    case "train":
      if (hasSelectedModel) return "done"
      return hasDataset ? "active" : "pending"
    case "validate":
      if (hasDeployment) return "done"
      return hasSelectedModel ? "active" : "pending"
    case "deploy":
      return hasDeployment ? "done" : hasSelectedModel ? "active" : "pending"
    default:
      return "pending"
  }
}

export function WorkflowProgress({
  hasDataset,
  hasSelectedModel,
  hasDeployment,
  onStepClick,
}: WorkflowProgressProps) {
  return (
    <div
      className="flex items-center gap-0 px-4 py-2.5 border-b bg-muted/30"
      data-testid="workflow-progress"
    >
      {STEPS.map((step, index) => {
        const status = stepStatus(step.key, hasDataset, hasSelectedModel, hasDeployment)
        const isLast = index === STEPS.length - 1

        return (
          <div key={step.key} className="flex items-center flex-1 min-w-0">
            {/* Step pill */}
            <button
              onClick={() => onStepClick?.(step.tab)}
              disabled={status === "pending"}
              title={step.description}
              data-testid={`workflow-step-${step.key}`}
              data-status={status}
              className={`
                flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium
                transition-all whitespace-nowrap
                ${status === "done"
                  ? "bg-primary/10 text-primary cursor-pointer hover:bg-primary/20"
                  : status === "active"
                    ? "bg-primary text-primary-foreground cursor-pointer shadow-sm"
                    : "bg-transparent text-muted-foreground/50 cursor-default"
                }
              `}
            >
              {/* Status icon */}
              {status === "done" ? (
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="size-3 shrink-0"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth={2.5}
                >
                  <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                </svg>
              ) : status === "active" ? (
                <span className="size-1.5 shrink-0 rounded-full bg-current" />
              ) : (
                <span className="size-1.5 shrink-0 rounded-full bg-muted-foreground/30" />
              )}
              <span>{step.label}</span>
            </button>

            {/* Connector line between steps */}
            {!isLast && (
              <div
                className={`h-px flex-1 mx-1 transition-colors ${
                  stepStatus(
                    STEPS[index + 1].key,
                    hasDataset,
                    hasSelectedModel,
                    hasDeployment
                  ) !== "pending"
                    ? "bg-primary/30"
                    : "bg-border"
                }`}
              />
            )}
          </div>
        )
      })}
    </div>
  )
}
