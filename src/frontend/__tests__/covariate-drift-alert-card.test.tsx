import { render, screen } from "@testing-library/react"
import { CovariateDriftAlertCard } from "@/components/deploy/covariate-drift-alert-card"
import type { CovariateDriftAlertResult, CovariateDriftFeatureAlert } from "@/lib/types"

const numericAlert: CovariateDriftFeatureAlert = {
  feature: "revenue",
  feature_type: "numeric",
  severity: "high",
  description: "40% of values fall outside training range [100, 300].",
  oor_count: 2,
  oor_pct: 40.0,
  total_count: 5,
  train_min: 100,
  train_max: 300,
}

const categoricalAlert: CovariateDriftFeatureAlert = {
  feature: "region",
  feature_type: "categorical",
  severity: "medium",
  description: "20% of values are unseen categories.",
  unseen_count: 1,
  unseen_pct: 20.0,
  total_count: 5,
}

const lowResult: CovariateDriftAlertResult = {
  deployment_id: "dep-1",
  has_alerts: false,
  severity: "low",
  severity_label: "No Significant Drift",
  sample_count: 50,
  feature_count: 3,
  alert_count: 0,
  alerts: [],
  summary: "No significant input drift detected across 3 features.",
}

const mediumResult: CovariateDriftAlertResult = {
  deployment_id: "dep-1",
  has_alerts: true,
  severity: "medium",
  severity_label: "Some Drift",
  sample_count: 5,
  feature_count: 2,
  alert_count: 1,
  alerts: [categoricalAlert],
  summary: "1 feature shows moderate drift in the last 5 predictions.",
}

const highResult: CovariateDriftAlertResult = {
  deployment_id: "dep-1",
  has_alerts: true,
  severity: "high",
  severity_label: "Significant Drift",
  sample_count: 5,
  feature_count: 2,
  alert_count: 1,
  alerts: [numericAlert],
  summary: "1 feature shows significant drift in the last 5 predictions.",
}

const multiAlertResult: CovariateDriftAlertResult = {
  deployment_id: "dep-1",
  has_alerts: true,
  severity: "high",
  severity_label: "Significant Drift",
  sample_count: 5,
  feature_count: 2,
  alert_count: 2,
  alerts: [numericAlert, categoricalAlert],
  summary: "2 features show drift in the last 5 predictions.",
}

describe("CovariateDriftAlertCard — rendering", () => {
  it("renders accessible figure with aria-label", () => {
    render(<CovariateDriftAlertCard result={lowResult} />)
    expect(screen.getByRole("figure", { name: /covariate drift alert/i })).toBeInTheDocument()
  })

  it("shows 'Production Input Drift Check' heading", () => {
    render(<CovariateDriftAlertCard result={lowResult} />)
    expect(screen.getByText("Production Input Drift Check")).toBeInTheDocument()
  })

  it("shows severity label for low severity", () => {
    render(<CovariateDriftAlertCard result={lowResult} />)
    expect(screen.getByText("No Significant Drift")).toBeInTheDocument()
  })

  it("shows severity label for medium severity", () => {
    render(<CovariateDriftAlertCard result={mediumResult} />)
    expect(screen.getByText("Some Drift")).toBeInTheDocument()
  })

  it("shows severity label for high severity", () => {
    render(<CovariateDriftAlertCard result={highResult} />)
    expect(screen.getByText("Significant Drift")).toBeInTheDocument()
  })

  it("shows sample count badge", () => {
    render(<CovariateDriftAlertCard result={lowResult} />)
    expect(screen.getByText("50 predictions analyzed")).toBeInTheDocument()
  })

  it("shows singular 'prediction' for sample_count=1", () => {
    const singleSample = { ...lowResult, sample_count: 1 }
    render(<CovariateDriftAlertCard result={singleSample} />)
    expect(screen.getByText("1 prediction analyzed")).toBeInTheDocument()
  })

  it("shows feature count badge when alerts present", () => {
    render(<CovariateDriftAlertCard result={mediumResult} />)
    expect(screen.getByText("1 feature flagged")).toBeInTheDocument()
  })

  it("shows plural 'features' badge for multiple alerts", () => {
    render(<CovariateDriftAlertCard result={multiAlertResult} />)
    expect(screen.getByText("2 features flagged")).toBeInTheDocument()
  })

  it("does not show features-flagged badge when no alerts", () => {
    render(<CovariateDriftAlertCard result={lowResult} />)
    expect(screen.queryByText(/features? flagged/i)).not.toBeInTheDocument()
  })

  it("shows summary text", () => {
    render(<CovariateDriftAlertCard result={lowResult} />)
    expect(screen.getByText(lowResult.summary)).toBeInTheDocument()
  })

  it("renders no alert rows for low severity", () => {
    render(<CovariateDriftAlertCard result={lowResult} />)
    expect(screen.queryByRole("generic", { name: /drift alert for/i })).not.toBeInTheDocument()
  })
})

describe("CovariateDriftAlertCard — alert rows", () => {
  it("renders numeric alert row with feature name", () => {
    render(<CovariateDriftAlertCard result={highResult} />)
    expect(screen.getByText("revenue")).toBeInTheDocument()
  })

  it("shows out-of-range percentage for numeric alert", () => {
    render(<CovariateDriftAlertCard result={highResult} />)
    expect(screen.getByText("40% out-of-range")).toBeInTheDocument()
  })

  it("shows HIGH badge for high-severity alert", () => {
    render(<CovariateDriftAlertCard result={highResult} />)
    expect(screen.getByText("HIGH")).toBeInTheDocument()
  })

  it("renders categorical alert row with feature name", () => {
    render(<CovariateDriftAlertCard result={mediumResult} />)
    expect(screen.getByText("region")).toBeInTheDocument()
  })

  it("shows unseen percentage for categorical alert", () => {
    render(<CovariateDriftAlertCard result={mediumResult} />)
    expect(screen.getByText("20% unseen categories")).toBeInTheDocument()
  })

  it("shows MED badge for medium-severity alert", () => {
    render(<CovariateDriftAlertCard result={mediumResult} />)
    expect(screen.getByText("MED")).toBeInTheDocument()
  })

  it("renders both alert rows for multi-alert result", () => {
    render(<CovariateDriftAlertCard result={multiAlertResult} />)
    expect(screen.getByText("revenue")).toBeInTheDocument()
    expect(screen.getByText("region")).toBeInTheDocument()
  })

  it("shows alert description", () => {
    render(<CovariateDriftAlertCard result={highResult} />)
    expect(screen.getByText(numericAlert.description)).toBeInTheDocument()
  })

  it("alert row has accessible aria-label", () => {
    render(<CovariateDriftAlertCard result={highResult} />)
    expect(
      screen.getByRole("generic", { name: /drift alert for revenue/i })
    ).toBeInTheDocument()
  })
})

describe("CovariateDriftAlertCard — guidance footer", () => {
  it("shows guidance footer when has_alerts is true", () => {
    render(<CovariateDriftAlertCard result={highResult} />)
    expect(screen.getByText(/show production input distribution/i)).toBeInTheDocument()
  })

  it("does not show guidance footer when no alerts", () => {
    render(<CovariateDriftAlertCard result={lowResult} />)
    expect(screen.queryByText(/show production input distribution/i)).not.toBeInTheDocument()
  })

  it("sr-only caption includes severity label", () => {
    render(<CovariateDriftAlertCard result={highResult} />)
    expect(screen.getAllByText(/Significant Drift/i).length).toBeGreaterThanOrEqual(1)
  })
})
