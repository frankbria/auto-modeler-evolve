import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";
import { createRequire } from "node:module";

// The installed react, not the range in package.json. eslint-config-next sets
// settings.react.version = "detect", the only path that calls
// context.getFilename() — removed in ESLint 10. Pinning the version skips
// detection and keeps us compatible with a future ESLint 10 bump.
const reactVersion = createRequire(import.meta.url)("react/package.json").version;

const eslintConfig = defineConfig([
  ...nextVitals,
  ...nextTs,
  { settings: { react: { version: reactVersion } } },
  {
    // Project rule: icons come from Hugeicons only, never lucide-react.
    rules: {
      "no-restricted-imports": [
        "error",
        {
          paths: [
            {
              name: "lucide-react",
              message: "Use @hugeicons/react instead of lucide-react.",
            },
          ],
        },
      ],
    },
  },
  // Override default ignores of eslint-config-next.
  globalIgnores([
    // Default ignores of eslint-config-next:
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
  ]),
]);

export default eslintConfig;
