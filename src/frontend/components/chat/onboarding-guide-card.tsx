"use client"

import type { OnboardingGuideResult, OnboardingStep } from "@/lib/types"

interface OnboardingGuideCardProps {
  guide: OnboardingGuideResult
  onSwitchTab?: (tab: string) => void
}

export function OnboardingGuideCard({
  guide,
  onSwitchTab,
}: OnboardingGuideCardProps) {
  const { steps, current_step, is_complete, completion_pct, summary } = guide

  return (
    <figure
      className="mt-3 rounded-xl border border-blue-200 bg-blue-50 p-4 shadow-sm max-w-md"
      aria-label="Guided onboarding wizard"
    >
      {/* Header */}
      <div className="flex items-center gap-2 mb-3">
        <span className="text-xl" aria-hidden="true">
          🧭
        </span>
        <div className="flex-1 min-w-0">
          <p className="font-semibold text-blue-900 text-sm leading-tight">
            {is_complete ? "You're all set!" : "Getting Started Guide"}
          </p>
          <p className="text-xs text-blue-600">{summary}</p>
        </div>
        {/* Completion badge */}
        <span className="shrink-0 text-xs font-semibold px-2 py-0.5 rounded-full bg-blue-100 text-blue-800">
          {completion_pct}%
        </span>
      </div>

      {/* Progress bar */}
      <div
        className="h-1.5 rounded-full bg-blue-100 mb-3 overflow-hidden"
        role="progressbar"
        aria-valuenow={completion_pct}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`Onboarding progress: ${completion_pct}%`}
      >
        <div
          className="h-full rounded-full bg-blue-500 transition-all duration-300"
          style={{ width: `${completion_pct}%` }}
        />
      </div>

      {/* Step list */}
      <ol className="space-y-1.5 mb-3" aria-label="Onboarding steps">
        {steps.map((step: OnboardingStep) => (
          <li
            key={step.name}
            className={`flex items-start gap-2 text-xs rounded-lg px-2 py-1.5 ${
              step.is_current
                ? "bg-blue-100 border border-blue-300"
                : step.is_done
                  ? "text-blue-400"
                  : "text-blue-600/60"
            }`}
          >
            {/* Status icon */}
            <span
              className="mt-0.5 shrink-0 text-sm"
              aria-hidden="true"
            >
              {step.is_done ? "✓" : step.is_current ? step.icon : "○"}
            </span>
            <div className="flex-1 min-w-0">
              <p
                className={`font-medium leading-tight ${
                  step.is_current
                    ? "text-blue-900"
                    : step.is_done
                      ? "line-through text-blue-400"
                      : "text-blue-500"
                }`}
              >
                {step.title}
              </p>
              {step.is_current && (
                <p className="text-blue-700 mt-0.5 leading-snug">
                  {step.description}
                </p>
              )}
            </div>
          </li>
        ))}
      </ol>

      {/* Current step tip + CTA */}
      {current_step && !is_complete && (
        <div className="bg-white rounded-lg border border-blue-200 p-2.5">
          <p className="text-xs text-blue-700 italic mb-2">
            💡 {current_step.hint}
          </p>
          {current_step.suggested_tab && onSwitchTab && (
            <button
              onClick={() => onSwitchTab(current_step.suggested_tab!)}
              className="w-full text-center text-xs font-medium py-1.5 px-3 rounded-lg bg-blue-600 text-white hover:bg-blue-700 transition-colors"
              aria-label={`Go to ${current_step.suggested_action}`}
            >
              {current_step.suggested_action} →
            </button>
          )}
        </div>
      )}

      {is_complete && (
        <p className="text-xs text-blue-700 text-center font-medium">
          🎉 Your model is deployed and ready to share!
        </p>
      )}
    </figure>
  )
}
