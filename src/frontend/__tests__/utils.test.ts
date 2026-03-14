/**
 * Tests for lib/utils.ts — the cn() className merging utility.
 *
 * cn() is used throughout every UI component to compose Tailwind classes.
 * Correct behavior here affects all visual rendering.
 */

import { cn } from "../lib/utils"

describe("cn()", () => {
  it("merges two class strings", () => {
    expect(cn("foo", "bar")).toBe("foo bar")
  })

  it("deduplicates conflicting Tailwind classes (last wins)", () => {
    // Tailwind-merge: p-4 overrides p-2
    expect(cn("p-2", "p-4")).toBe("p-4")
  })

  it("handles conditional classes via clsx", () => {
    expect(cn("base", false && "hidden", "visible")).toBe("base visible")
    expect(cn("base", true && "active")).toBe("base active")
  })

  it("ignores undefined and null values", () => {
    expect(cn("a", undefined, null, "b")).toBe("a b")
  })

  it("returns empty string for no arguments", () => {
    expect(cn()).toBe("")
  })

  it("merges object syntax from clsx", () => {
    expect(cn({ "text-red-500": true, "text-blue-500": false })).toBe("text-red-500")
  })

  it("handles array syntax from clsx", () => {
    expect(cn(["foo", "bar"])).toBe("foo bar")
  })

  it("removes duplicate identical classes", () => {
    const result = cn("flex flex-col", "flex")
    // After merge, 'flex' appears once (tailwind-merge deduplicates; order may vary)
    const classes = result.split(" ")
    expect(classes.filter((c) => c === "flex")).toHaveLength(1)
    expect(classes).toContain("flex-col")
  })
})
