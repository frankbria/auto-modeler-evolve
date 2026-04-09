/**
 * Tests for PresetSavedCard and PresetListCard components.
 */

import React from "react"
import { render, screen, fireEvent } from "@testing-library/react"
import { PresetSavedCard } from "@/components/deploy/preset-saved-card"
import { PresetListCard } from "@/components/deploy/preset-list-card"
import type { PresetSavedInfo, PresetListInfo } from "@/lib/types"

const PRESET_SAVED: PresetSavedInfo = {
  id: "preset-1",
  deployment_id: "dep-1",
  name: "Best Case",
  feature_values: { units: 500, region: "East" },
  feature_count: 2,
}

const PRESET_LIST_DATA: PresetListInfo = {
  presets: [
    {
      id: "preset-1",
      name: "Best Case",
      feature_values: { units: 500, region: "East" },
      feature_count: 2,
    },
    {
      id: "preset-2",
      name: "Worst Case",
      feature_values: { units: 10, region: "West" },
      feature_count: 2,
    },
  ],
  count: 2,
  deployment_id: "dep-1",
}

// ---------------------------------------------------------------------------
// PresetSavedCard
// ---------------------------------------------------------------------------

describe("PresetSavedCard", () => {
  it("renders with data-testid", () => {
    render(<PresetSavedCard preset={PRESET_SAVED} />)
    expect(screen.getByTestId("preset-saved-card")).toBeTruthy()
  })

  it("has decorative icon aria-hidden", () => {
    render(<PresetSavedCard preset={PRESET_SAVED} />)
    const icon = screen.getByText("🎯")
    expect(icon).toHaveAttribute("aria-hidden", "true")
  })

  it("displays preset name", () => {
    render(<PresetSavedCard preset={PRESET_SAVED} />)
    expect(screen.getByText(/Best Case/)).toBeTruthy()
  })

  it("shows feature count badge", () => {
    render(<PresetSavedCard preset={PRESET_SAVED} />)
    expect(screen.getByText(/2 features/)).toBeTruthy()
  })

  it("renders feature value badges", () => {
    render(<PresetSavedCard preset={PRESET_SAVED} />)
    const badges = screen.getAllByTestId("preset-feature-badge")
    expect(badges.length).toBe(2)
  })

  it("shows units badge", () => {
    render(<PresetSavedCard preset={PRESET_SAVED} />)
    expect(screen.getByText("units=500")).toBeTruthy()
  })

  it("shows region badge", () => {
    render(<PresetSavedCard preset={PRESET_SAVED} />)
    expect(screen.getByText("region=East")).toBeTruthy()
  })

  it("includes VP dashboard hint", () => {
    render(<PresetSavedCard preset={PRESET_SAVED} />)
    expect(screen.getByText(/quick-fill button/i)).toBeTruthy()
  })

  it("uses singular feature for count of 1", () => {
    const singlePreset: PresetSavedInfo = {
      ...PRESET_SAVED,
      feature_values: { units: 100 },
      feature_count: 1,
    }
    render(<PresetSavedCard preset={singlePreset} />)
    expect(screen.getByText(/1 feature/)).toBeTruthy()
    expect(screen.queryByText(/1 features/)).toBeNull()
  })
})

// ---------------------------------------------------------------------------
// PresetListCard
// ---------------------------------------------------------------------------

describe("PresetListCard", () => {
  it("renders with accessible aria-label", () => {
    render(<PresetListCard preset_list={PRESET_LIST_DATA} />)
    expect(screen.getByTestId("preset-list-card")).toBeTruthy()
  })

  it("has decorative icon aria-hidden", () => {
    render(<PresetListCard preset_list={PRESET_LIST_DATA} />)
    expect(screen.getByText("📋")).toHaveAttribute("aria-hidden", "true")
  })

  it("shows count badge", () => {
    render(<PresetListCard preset_list={PRESET_LIST_DATA} />)
    expect(screen.getByText(/2 presets/)).toBeTruthy()
  })

  it("renders preset rows", () => {
    render(<PresetListCard preset_list={PRESET_LIST_DATA} />)
    const rows = screen.getAllByTestId("preset-list-row")
    expect(rows.length).toBe(2)
  })

  it("shows preset names", () => {
    render(<PresetListCard preset_list={PRESET_LIST_DATA} />)
    expect(screen.getByText("Best Case")).toBeTruthy()
    expect(screen.getByText("Worst Case")).toBeTruthy()
  })

  it("renders Load buttons when onLoadPreset provided", () => {
    const mockLoad = jest.fn()
    render(<PresetListCard preset_list={PRESET_LIST_DATA} onLoadPreset={mockLoad} />)
    const buttons = screen.getAllByTestId("preset-load-button")
    expect(buttons.length).toBe(2)
  })

  it("calls onLoadPreset with feature values on click", () => {
    const mockLoad = jest.fn()
    render(<PresetListCard preset_list={PRESET_LIST_DATA} onLoadPreset={mockLoad} />)
    const buttons = screen.getAllByTestId("preset-load-button")
    fireEvent.click(buttons[0])
    expect(mockLoad).toHaveBeenCalledWith(PRESET_LIST_DATA.presets[0].feature_values)
  })

  it("does not render Load buttons without onLoadPreset", () => {
    render(<PresetListCard preset_list={PRESET_LIST_DATA} />)
    expect(screen.queryByTestId("preset-load-button")).toBeNull()
  })

  it("shows empty state when no presets", () => {
    const empty: PresetListInfo = { presets: [], count: 0, deployment_id: "dep-1" }
    render(<PresetListCard preset_list={empty} />)
    expect(screen.getByText(/No presets saved yet/i)).toBeTruthy()
  })

  it("uses singular preset for count of 1", () => {
    const singleList: PresetListInfo = {
      presets: [PRESET_LIST_DATA.presets[0]],
      count: 1,
      deployment_id: "dep-1",
    }
    render(<PresetListCard preset_list={singleList} />)
    expect(screen.getByText(/1 preset/)).toBeTruthy()
    expect(screen.queryByText(/1 presets/)).toBeNull()
  })

  it("shows store action is connected (store exports)", async () => {
    // Test that the store exports the expected action
    const { useAppStore } = await import("@/lib/store")
    const state = useAppStore.getState()
    expect(typeof state.attachPresetSavedToLastMessage).toBe("function")
    expect(typeof state.attachPresetListToLastMessage).toBe("function")
  })
})
