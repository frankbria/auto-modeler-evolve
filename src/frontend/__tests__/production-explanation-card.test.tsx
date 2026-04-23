import React from "react"
import { render, screen } from "@testing-library/react"
import { ProductionExplanationCard } from "@/components/chat/production-explanation-card"
import type { ProdPredictionExplanationResult } from "@/lib/types"
import { useAppStore } from "@/lib/store"

const regressionResult: ProdPredictionExplanationResult = {
  prediction_log_id: "log-abc-123",
  created_at: "2026-04-23T04:00:00Z",
  prediction: 42500,
  confidence: null,
  algorithm: "random_forest_regressor",
  target_column: "revenue",
  problem_type: "regression",
  deployment_id: "dep-xyz",
  contributions: [
    { feature: "units", value: 150, mean_value: 100, contribution: 0.45, direction: "positive" },
    { feature: "price", value: 25, mean_value: 20, contribution: 0.22, direction: "positive" },
    { feature: "region", value: 1, mean_value: 0.5, contribution: -0.15, direction: "negative" },
  ],
  top_drivers: ["units", "price", "region"],
  summary: "The prediction of 42500 was primarily driven by units and price.",
}

const classificationResult: ProdPredictionExplanationResult = {
  prediction_log_id: "log-def-456",
  created_at: "2026-04-23T10:30:00Z",
  prediction: "churned",
  confidence: 0.87,
  algorithm: "logistic_regression",
  target_column: "churn",
  problem_type: "classification",
  deployment_id: "dep-abc",
  contributions: [
    { feature: "tenure", value: 2, mean_value: 24, contribution: -0.60, direction: "negative" },
    { feature: "monthly_charges", value: 95, mean_value: 65, contribution: 0.35, direction: "positive" },
  ],
  top_drivers: ["tenure", "monthly_charges"],
  summary: "The model predicted churn because tenure was far below average.",
}

describe("ProductionExplanationCard", () => {
  describe("basic structure", () => {
    it("renders heading", () => {
      render(<ProductionExplanationCard result={regressionResult} />)
      expect(screen.getByText("Production Prediction Explained")).toBeInTheDocument()
    })

    it("has aria label for the region", () => {
      render(<ProductionExplanationCard result={regressionResult} />)
      expect(screen.getByRole("region", { name: /production prediction explanation/i })).toBeInTheDocument()
    })

    it("shows algorithm badge", () => {
      render(<ProductionExplanationCard result={regressionResult} />)
      expect(screen.getByText("Random Forest")).toBeInTheDocument()
    })

    it("shows problem type badge (regression)", () => {
      render(<ProductionExplanationCard result={regressionResult} />)
      expect(screen.getByText("Regression")).toBeInTheDocument()
    })

    it("shows problem type badge (classification)", () => {
      render(<ProductionExplanationCard result={classificationResult} />)
      expect(screen.getByText("Classification")).toBeInTheDocument()
    })
  })

  describe("prediction value", () => {
    it("shows target column label", () => {
      render(<ProductionExplanationCard result={regressionResult} />)
      expect(screen.getByText("revenue:")).toBeInTheDocument()
    })

    it("shows prediction value", () => {
      render(<ProductionExplanationCard result={regressionResult} />)
      expect(screen.getByText("42500")).toBeInTheDocument()
    })

    it("shows classification prediction", () => {
      render(<ProductionExplanationCard result={classificationResult} />)
      expect(screen.getByText("churned")).toBeInTheDocument()
    })

    it("shows confidence badge when available", () => {
      render(<ProductionExplanationCard result={classificationResult} />)
      expect(screen.getByText("87% confidence")).toBeInTheDocument()
    })

    it("hides confidence when null", () => {
      render(<ProductionExplanationCard result={regressionResult} />)
      expect(screen.queryByText(/confidence/i)).not.toBeInTheDocument()
    })
  })

  describe("feature contributions", () => {
    it("renders contribution rows with list role", () => {
      render(<ProductionExplanationCard result={regressionResult} />)
      expect(screen.getByRole("list", { name: /feature contributions/i })).toBeInTheDocument()
    })

    it("renders each feature name", () => {
      render(<ProductionExplanationCard result={regressionResult} />)
      expect(screen.getByText("units")).toBeInTheDocument()
      expect(screen.getByText("price")).toBeInTheDocument()
      expect(screen.getByText("region")).toBeInTheDocument()
    })

    it("shows positive contribution with + sign", () => {
      render(<ProductionExplanationCard result={regressionResult} />)
      expect(screen.getByText("+0.450")).toBeInTheDocument()
    })

    it("shows negative contribution without + sign", () => {
      render(<ProductionExplanationCard result={regressionResult} />)
      expect(screen.getByText("-0.150")).toBeInTheDocument()
    })

    it("shows feature value annotations", () => {
      render(<ProductionExplanationCard result={regressionResult} />)
      expect(screen.getByText("val: 150")).toBeInTheDocument()
    })

    it("renders aria labels per contribution row", () => {
      render(<ProductionExplanationCard result={regressionResult} />)
      expect(screen.getByRole("listitem", { name: /units:.*contribution/i })).toBeInTheDocument()
    })
  })

  describe("summary", () => {
    it("shows the summary sentence", () => {
      render(<ProductionExplanationCard result={regressionResult} />)
      expect(
        screen.getByText("The prediction of 42500 was primarily driven by units and price.")
      ).toBeInTheDocument()
    })
  })

  describe("accessibility", () => {
    it("has a sr-only figcaption", () => {
      const { container } = render(<ProductionExplanationCard result={regressionResult} />)
      const figcaption = container.querySelector("figcaption")
      expect(figcaption).toBeInTheDocument()
      expect(figcaption?.className).toContain("sr-only")
    })

    it("figcaption mentions target column", () => {
      const { container } = render(<ProductionExplanationCard result={regressionResult} />)
      const fc = container.querySelector("figcaption")
      expect(fc?.textContent).toContain("revenue")
    })
  })

  describe("store action", () => {
    it("store exports attachProdPredictionExplanationToLastMessage", () => {
      const state = useAppStore.getState()
      expect(typeof state.attachProdPredictionExplanationToLastMessage).toBe("function")
    })

    it("action attaches result to last assistant message", () => {
      useAppStore.setState({
        messages: [
          { id: "1", role: "user", content: "explain" },
          { id: "2", role: "assistant", content: "Here's the explanation..." },
        ],
      })
      useAppStore.getState().attachProdPredictionExplanationToLastMessage(regressionResult)
      const msgs = useAppStore.getState().messages
      expect(msgs[1].prod_prediction_explanation).toEqual(regressionResult)
    })

    it("does not attach to user messages", () => {
      useAppStore.setState({
        messages: [{ id: "1", role: "user", content: "explain" }],
      })
      useAppStore.getState().attachProdPredictionExplanationToLastMessage(regressionResult)
      const msgs = useAppStore.getState().messages
      expect(msgs[0].prod_prediction_explanation).toBeUndefined()
    })
  })
})
