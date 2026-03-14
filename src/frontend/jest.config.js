const nextJest = require("next/jest")

const createJestConfig = nextJest({
  dir: "./",
})

/** @type {import('jest').Config} */
const config = {
  coverageProvider: "v8",
  testEnvironment: "jsdom",
  setupFilesAfterEnv: ["<rootDir>/jest.setup.ts"],
  testMatch: ["**/__tests__/**/*.[jt]s?(x)", "**/*.test.[jt]s?(x)"],
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/$1",
  },
  coveragePathIgnorePatterns: ["/node_modules/", "/.next/"],
  // Collect coverage from all source files so untested components appear at 0%
  collectCoverageFrom: [
    "components/**/*.{ts,tsx}",
    "lib/**/*.{ts,tsx}",
    "app/**/*.{ts,tsx}",
    "!**/*.d.ts",
    "!**/ui/**",
    // types.ts is pure TypeScript interface declarations — no runtime code to cover
    "!lib/types.ts",
    // layout.tsx is a Next.js root layout — not meaningfully unit-testable
    "!app/layout.tsx",
  ],
  // Exclude Playwright E2E tests from Jest
  testPathIgnorePatterns: ["/node_modules/", "/.next/", "/e2e/"],
}

module.exports = createJestConfig(config)
