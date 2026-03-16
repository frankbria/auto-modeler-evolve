import { render, screen, fireEvent } from "@testing-library/react"
import { WorkflowProgress } from "@/components/ui/workflow-progress"

describe("WorkflowProgress", () => {
  const defaultProps = {
    hasDataset: false,
    hasSelectedModel: false,
    hasDeployment: false,
    onStepClick: jest.fn(),
  }

  beforeEach(() => {
    jest.clearAllMocks()
  })

  it("renders all 4 steps", () => {
    render(<WorkflowProgress {...defaultProps} />)
    expect(screen.getByTestId("workflow-step-upload")).toBeInTheDocument()
    expect(screen.getByTestId("workflow-step-train")).toBeInTheDocument()
    expect(screen.getByTestId("workflow-step-validate")).toBeInTheDocument()
    expect(screen.getByTestId("workflow-step-deploy")).toBeInTheDocument()
  })

  it("shows upload as active when no dataset", () => {
    render(<WorkflowProgress {...defaultProps} hasDataset={false} />)
    expect(screen.getByTestId("workflow-step-upload")).toHaveAttribute("data-status", "active")
    expect(screen.getByTestId("workflow-step-train")).toHaveAttribute("data-status", "pending")
    expect(screen.getByTestId("workflow-step-validate")).toHaveAttribute("data-status", "pending")
    expect(screen.getByTestId("workflow-step-deploy")).toHaveAttribute("data-status", "pending")
  })

  it("shows upload as done and train as active when dataset exists", () => {
    render(<WorkflowProgress {...defaultProps} hasDataset={true} />)
    expect(screen.getByTestId("workflow-step-upload")).toHaveAttribute("data-status", "done")
    expect(screen.getByTestId("workflow-step-train")).toHaveAttribute("data-status", "active")
    expect(screen.getByTestId("workflow-step-validate")).toHaveAttribute("data-status", "pending")
    expect(screen.getByTestId("workflow-step-deploy")).toHaveAttribute("data-status", "pending")
  })

  it("shows train as done and validate as active when model is selected", () => {
    render(<WorkflowProgress {...defaultProps} hasDataset={true} hasSelectedModel={true} />)
    expect(screen.getByTestId("workflow-step-upload")).toHaveAttribute("data-status", "done")
    expect(screen.getByTestId("workflow-step-train")).toHaveAttribute("data-status", "done")
    expect(screen.getByTestId("workflow-step-validate")).toHaveAttribute("data-status", "active")
    expect(screen.getByTestId("workflow-step-deploy")).toHaveAttribute("data-status", "active")
  })

  it("shows all steps as done when fully deployed", () => {
    render(
      <WorkflowProgress
        {...defaultProps}
        hasDataset={true}
        hasSelectedModel={true}
        hasDeployment={true}
      />
    )
    expect(screen.getByTestId("workflow-step-upload")).toHaveAttribute("data-status", "done")
    expect(screen.getByTestId("workflow-step-train")).toHaveAttribute("data-status", "done")
    expect(screen.getByTestId("workflow-step-validate")).toHaveAttribute("data-status", "done")
    expect(screen.getByTestId("workflow-step-deploy")).toHaveAttribute("data-status", "done")
  })

  it("calls onStepClick with correct tab when clicking a done step", () => {
    const onStepClick = jest.fn()
    render(
      <WorkflowProgress
        {...defaultProps}
        hasDataset={true}
        onStepClick={onStepClick}
      />
    )
    fireEvent.click(screen.getByTestId("workflow-step-upload"))
    expect(onStepClick).toHaveBeenCalledWith("data")
  })

  it("calls onStepClick for active train step", () => {
    const onStepClick = jest.fn()
    render(
      <WorkflowProgress {...defaultProps} hasDataset={true} onStepClick={onStepClick} />
    )
    fireEvent.click(screen.getByTestId("workflow-step-train"))
    expect(onStepClick).toHaveBeenCalledWith("models")
  })

  it("does not call onStepClick when clicking a pending step (disabled)", () => {
    const onStepClick = jest.fn()
    render(<WorkflowProgress {...defaultProps} hasDataset={false} onStepClick={onStepClick} />)
    // train/validate/deploy are pending when no dataset
    fireEvent.click(screen.getByTestId("workflow-step-train"))
    expect(onStepClick).not.toHaveBeenCalled()
  })

  it("renders without onStepClick prop (optional)", () => {
    // Should not throw
    render(<WorkflowProgress hasDataset={true} hasSelectedModel={false} hasDeployment={false} />)
    expect(screen.getByTestId("workflow-progress")).toBeInTheDocument()
  })

  it("renders the container with correct testid", () => {
    render(<WorkflowProgress {...defaultProps} />)
    expect(screen.getByTestId("workflow-progress")).toBeInTheDocument()
  })
})
