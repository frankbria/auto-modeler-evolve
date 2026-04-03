import "@testing-library/jest-dom"

// jest-environment-jsdom doesn't expose TextDecoder/TextEncoder as globals.
// Polyfill from Node.js 'util' so SSE stream decoding works in tests.
// eslint-disable-next-line @typescript-eslint/no-require-imports
const { TextDecoder: NodeTextDecoder, TextEncoder: NodeTextEncoder } = require("util")
global.TextDecoder = NodeTextDecoder
global.TextEncoder = NodeTextEncoder

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

// @base-ui/react ScrollArea uses getAnimations() which is not implemented in jsdom.
// Stub it so components using ScrollArea don't crash in tests.
if (!Element.prototype.getAnimations) {
  Element.prototype.getAnimations = () => []
}

// jsdom doesn't implement scrollIntoView — stub it so refs that call it don't crash.
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = () => {}
}
