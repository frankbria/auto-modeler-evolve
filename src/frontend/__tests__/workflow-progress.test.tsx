import { render, screen, fireEvent } from "@testing-library/react"
import { WorkflowProgress } from "@/components/ui/workflow-progress"

describe("WorkflowProgress", () => {
  const defaultProps = {
    hasDataset: false,
    hasFeatures: false,
    hasSelectedModel: false,
    hasValidation: false,
    hasDeployment: false,
    onStepClick: jest.fn(),
  }

  beforeEach(() => {
    jest.clearAllMocks()
  })

  it("renders all 5 steps", () => {
    render(<WorkflowProgress {...defaultProps} />)
    expect(screen.getByTestId("workflow-step-upload")).toBeInTheDocument()
    expect(screen.getByTestId("workflow-step-features")).toBeInTheDocument()
    expect(screen.getByTestId("workflow-step-train")).toBeInTheDocument()
    expect(screen.getByTestId("workflow-step-validate")).toBeInTheDocument()
    expect(screen.getByTestId("workflow-step-deploy")).toBeInTheDocument()
  })

  it("shows upload as active when no dataset", () => {
    render(<WorkflowProgress {...defaultProps} hasDataset={false} />)
    expect(screen.getByTestId("workflow-step-upload")).toHaveAttribute("data-status", "active")
    expect(screen.getByTestId("workflow-step-features")).toHaveAttribute("data-status", "pending")
    expect(screen.getByTestId("workflow-step-train")).toHaveAttribute("data-status", "pending")
    expect(screen.getByTestId("workflow-step-validate")).toHaveAttribute("data-status", "pending")
    expect(screen.getByTestId("workflow-step-deploy")).toHaveAttribute("data-status", "pending")
  })

  it("shows upload as done and features as active when dataset exists", () => {
    render(<WorkflowProgress {...defaultProps} hasDataset={true} />)
    expect(screen.getByTestId("workflow-step-upload")).toHaveAttribute("data-status", "done")
    expect(screen.getByTestId("workflow-step-features")).toHaveAttribute("data-status", "active")
    expect(screen.getByTestId("workflow-step-train")).toHaveAttribute("data-status", "active")
    expect(screen.getByTestId("workflow-step-validate")).toHaveAttribute("data-status", "pending")
    expect(screen.getByTestId("workflow-step-deploy")).toHaveAttribute("data-status", "pending")
  })

  it("shows features as done and train as active when features applied", () => {
    render(<WorkflowProgress {...defaultProps} hasDataset={true} hasFeatures={true} />)
    expect(screen.getByTestId("workflow-step-upload")).toHaveAttribute("data-status", "done")
    expect(screen.getByTestId("workflow-step-features")).toHaveAttribute("data-status", "done")
    expect(screen.getByTestId("workflow-step-train")).toHaveAttribute("data-status", "active")
  })

  it("shows train as done and validate as active when model is selected", () => {
    render(<WorkflowProgress {...defaultProps} hasDataset={true} hasFeatures={true} hasSelectedModel={true} />)
    expect(screen.getByTestId("workflow-step-train")).toHaveAttribute("data-status", "done")
    expect(screen.getByTestId("workflow-step-validate")).toHaveAttribute("data-status", "active")
  })

  it("shows validate as done when validation results are present (not just on deployment)", () => {
    render(
      <WorkflowProgress
        {...defaultProps}
        hasDataset={true}
        hasFeatures={true}
        hasSelectedModel={true}
        hasValidation={true}
        hasDeployment={false}
      />
    )
    expect(screen.getByTestId("workflow-step-validate")).toHaveAttribute("data-status", "done")
    expect(screen.getByTestId("workflow-step-deploy")).toHaveAttribute("data-status", "active")
  })

  it("shows all steps as done when fully deployed with validation", () => {
    render(
      <WorkflowProgress
        {...defaultProps}
        hasDataset={true}
        hasFeatures={true}
        hasSelectedModel={true}
        hasValidation={true}
        hasDeployment={true}
      />
    )
    expect(screen.getByTestId("workflow-step-upload")).toHaveAttribute("data-status", "done")
    expect(screen.getByTestId("workflow-step-features")).toHaveAttribute("data-status", "done")
    expect(screen.getByTestId("workflow-step-train")).toHaveAttribute("data-status", "done")
    expect(screen.getByTestId("workflow-step-validate")).toHaveAttribute("data-status", "done")
    expect(screen.getByTestId("workflow-step-deploy")).toHaveAttribute("data-status", "done")
  })

  it("shows deploy as active when model selected but no validation/deployment", () => {
    render(
      <WorkflowProgress
        {...defaultProps}
        hasDataset={true}
        hasSelectedModel={true}
      />
    )
    expect(screen.getByTestId("workflow-step-deploy")).toHaveAttribute("data-status", "active")
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

  it("calls onStepClick with features tab for features step", () => {
    const onStepClick = jest.fn()
    render(<WorkflowProgress {...defaultProps} hasDataset={true} onStepClick={onStepClick} />)
    fireEvent.click(screen.getByTestId("workflow-step-features"))
    expect(onStepClick).toHaveBeenCalledWith("features")
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
    fireEvent.click(screen.getByTestId("workflow-step-train"))
    expect(onStepClick).not.toHaveBeenCalled()
  })

  it("renders without onStepClick prop (optional)", () => {
    render(<WorkflowProgress hasDataset={true} hasFeatures={false} hasSelectedModel={false} hasValidation={false} hasDeployment={false} />)
    expect(screen.getByTestId("workflow-progress")).toBeInTheDocument()
  })

  it("renders the container with correct testid", () => {
    render(<WorkflowProgress {...defaultProps} />)
    expect(screen.getByTestId("workflow-progress")).toBeInTheDocument()
  })
})
