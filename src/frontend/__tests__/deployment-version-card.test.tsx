/**
 * Tests for DeploymentVersionCard component.
 */

import React from "react"
import { render, screen, fireEvent, waitFor } from "@testing-library/react"
import { DeploymentVersionCard } from "@/components/deploy/deployment-version-card"
import { api } from "@/lib/api"
import type { DeploymentVersionHistory, RollbackResult } from "@/lib/types"

jest.mock("@/lib/api")

const mockHistory1Version: DeploymentVersionHistory = {
  deployment_id: "dep-1",
  current_version_number: 1,
  versions: [
    {
      id: "v1-id",
      deployment_id: "dep-1",
      version_number: 1,
      model_run_id: "run-1",
      algorithm: "linear_regression",
      problem_type: "regression",
      target_column: "revenue",
      metrics: { r2: 0.82, mae: 12.5 },
      pipeline_path: "/deployments/run-1_pipeline.joblib",
      deployed_at: "2024-03-01T09:00:00",
      is_current: true,
    },
  ],
}

const mockHistory2Versions: DeploymentVersionHistory = {
  deployment_id: "dep-1",
  current_version_number: 2,
  versions: [
    {
      id: "v2-id",
      deployment_id: "dep-1",
      version_number: 2,
      model_run_id: "run-2",
      algorithm: "random_forest",
      problem_type: "regression",
      target_column: "revenue",
      metrics: { r2: 0.89 },
      pipeline_path: "/deployments/run-2_pipeline.joblib",
      deployed_at: "2024-03-10T14:00:00",
      is_current: true,
    },
    {
      id: "v1-id",
      deployment_id: "dep-1",
      version_number: 1,
      model_run_id: "run-1",
      algorithm: "linear_regression",
      problem_type: "regression",
      target_column: "revenue",
      metrics: { r2: 0.82 },
      pipeline_path: "/deployments/run-1_pipeline.joblib",
      deployed_at: "2024-03-01T09:00:00",
      is_current: false,
    },
  ],
}

const mockRollbackResult: RollbackResult = {
  rolled_back_to_version: 1,
  new_version_number: 3,
  id: "dep-1",
  model_run_id: "run-1",
  endpoint_path: "/api/predict/dep-1",
  algorithm: "linear_regression",
  metrics: { r2: 0.82 },
  api_key_enabled: false,
}

describe("DeploymentVersionCard", () => {
  beforeEach(() => {
    jest.clearAllMocks()
    ;(api.deploy.getVersions as jest.Mock).mockResolvedValue(mockHistory2Versions)
    ;(api.deploy.rollback as jest.Mock).mockResolvedValue(mockRollbackResult)
  })

  it("renders nothing when only 1 version exists", async () => {
    ;(api.deploy.getVersions as jest.Mock).mockResolvedValue(mockHistory1Version)
    const { container } = render(
      <DeploymentVersionCard deploymentId="dep-1" />
    )
    await waitFor(() => {
      expect(api.deploy.getVersions).toHaveBeenCalledWith("dep-1")
    })
    expect(container.firstChild).toBeNull()
  })

  it("renders version history when 2+ versions exist", async () => {
    render(<DeploymentVersionCard deploymentId="dep-1" />)
    await waitFor(() => {
      expect(screen.getByTestId("deployment-version-card")).toBeInTheDocument()
    })
    expect(screen.getByText("Version History")).toBeInTheDocument()
    expect(screen.getByTestId("version-row-2")).toBeInTheDocument()
    expect(screen.getByTestId("version-row-1")).toBeInTheDocument()
  })

  it("shows Current badge on the active version", async () => {
    render(<DeploymentVersionCard deploymentId="dep-1" />)
    await waitFor(() => screen.getByTestId("deployment-version-card"))
    expect(screen.getByText("Current")).toBeInTheDocument()
  })

  it("shows algorithm names in version rows", async () => {
    render(<DeploymentVersionCard deploymentId="dep-1" />)
    await waitFor(() => screen.getByTestId("deployment-version-card"))
    expect(screen.getByText("Random Forest")).toBeInTheDocument()
    expect(screen.getByText("Linear Regression")).toBeInTheDocument()
  })

  it("shows Restore button on non-current versions", async () => {
    render(<DeploymentVersionCard deploymentId="dep-1" />)
    await waitFor(() => screen.getByTestId("deployment-version-card"))
    expect(screen.getByTestId("rollback-btn-1")).toBeInTheDocument()
    expect(screen.queryByTestId("rollback-btn-2")).toBeNull()
  })

  it("shows confirmation dialog on first Restore click", async () => {
    render(<DeploymentVersionCard deploymentId="dep-1" />)
    await waitFor(() => screen.getByTestId("deployment-version-card"))

    fireEvent.click(screen.getByTestId("rollback-btn-1"))
    expect(screen.getByTestId("rollback-confirm")).toBeInTheDocument()
    expect(screen.getByTestId("confirm-rollback-btn")).toBeInTheDocument()
    expect(screen.getByTestId("cancel-rollback-btn")).toBeInTheDocument()
  })

  it("calls api.deploy.rollback on confirm click and refreshes history", async () => {
    const onRollback = jest.fn()
    // After rollback, return updated history with v3
    const updatedHistory: DeploymentVersionHistory = {
      deployment_id: "dep-1",
      current_version_number: 3,
      versions: [
        { ...mockHistory2Versions.versions[1], version_number: 3, is_current: true },
        { ...mockHistory2Versions.versions[0], is_current: false },
        { ...mockHistory2Versions.versions[1], is_current: false },
      ],
    }
    ;(api.deploy.getVersions as jest.Mock)
      .mockResolvedValueOnce(mockHistory2Versions)
      .mockResolvedValueOnce(updatedHistory)

    render(<DeploymentVersionCard deploymentId="dep-1" onRollback={onRollback} />)
    await waitFor(() => screen.getByTestId("deployment-version-card"))

    fireEvent.click(screen.getByTestId("rollback-btn-1"))
    fireEvent.click(screen.getByTestId("confirm-rollback-btn"))

    await waitFor(() => {
      expect(api.deploy.rollback).toHaveBeenCalledWith("dep-1", 1)
    })
    expect(onRollback).toHaveBeenCalledWith("run-1")
  })

  it("cancels rollback when Cancel is clicked", async () => {
    render(<DeploymentVersionCard deploymentId="dep-1" />)
    await waitFor(() => screen.getByTestId("deployment-version-card"))

    fireEvent.click(screen.getByTestId("rollback-btn-1"))
    expect(screen.getByTestId("rollback-confirm")).toBeInTheDocument()

    fireEvent.click(screen.getByTestId("cancel-rollback-btn"))
    expect(screen.queryByTestId("rollback-confirm")).toBeNull()
  })

  it("shows error message when rollback fails", async () => {
    ;(api.deploy.rollback as jest.Mock).mockRejectedValue(new Error("HTTP 400"))
    render(<DeploymentVersionCard deploymentId="dep-1" />)
    await waitFor(() => screen.getByTestId("deployment-version-card"))

    fireEvent.click(screen.getByTestId("rollback-btn-1"))
    fireEvent.click(screen.getByTestId("confirm-rollback-btn"))

    await waitFor(() => {
      expect(screen.getByTestId("rollback-error")).toBeInTheDocument()
    })
    expect(screen.getByText(/HTTP 400/)).toBeInTheDocument()
  })

  it("shows R² metric value in version rows", async () => {
    render(<DeploymentVersionCard deploymentId="dep-1" />)
    await waitFor(() => screen.getByTestId("deployment-version-card"))
    // v2 has R² 0.89
    expect(getAllByText(screen.getByTestId("version-row-2"), /0\.89/)).toBeTruthy()
  })

  it("shows fallback metric for classification deployments", async () => {
    const classHistory: DeploymentVersionHistory = {
      ...mockHistory2Versions,
      versions: mockHistory2Versions.versions.map((v) => ({
        ...v,
        metrics: { accuracy: 0.92 },
        problem_type: "classification",
      })),
    }
    ;(api.deploy.getVersions as jest.Mock).mockResolvedValue(classHistory)
    render(<DeploymentVersionCard deploymentId="dep-1" />)
    await waitFor(() => screen.getByTestId("deployment-version-card"))
    const accuracyText = screen.getAllByText(/92\.0%/)
    expect(accuracyText.length > 0).toBe(true)
  })

  it("shows current version number in card header", async () => {
    render(<DeploymentVersionCard deploymentId="dep-1" />)
    await waitFor(() => screen.getByTestId("deployment-version-card"))
    expect(screen.getByText(/Current: v2/)).toBeInTheDocument()
  })

  it("calls getVersions with the deployment ID on mount", async () => {
    render(<DeploymentVersionCard deploymentId="dep-42" />)
    await waitFor(() => {
      expect(api.deploy.getVersions).toHaveBeenCalledWith("dep-42")
    })
  })
})

// Helper to query within a container element
function getAllByText(container: HTMLElement, matcher: RegExp): HTMLElement[] {
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT)
  const found: HTMLElement[] = []
  let node: Node | null = walker.nextNode()
  while (node) {
    if (node.textContent && matcher.test(node.textContent)) {
      found.push(node.parentElement as HTMLElement)
    }
    node = walker.nextNode()
  }
  return found
}
