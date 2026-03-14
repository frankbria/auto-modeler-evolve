import "@testing-library/jest-dom"

// Recharts uses ResizeObserver internally (via ResponsiveContainer).
// jsdom doesn't implement it, so we stub it out.
global.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

// navigator.clipboard is not available in jsdom
Object.assign(navigator, {
  clipboard: {
    writeText: jest.fn().mockResolvedValue(undefined),
    readText: jest.fn().mockResolvedValue(""),
  },
})
