import { BACKEND } from "./env"

/**
 * Preflight: confirm the server answering on BACKEND is actually AutoModeler.
 *
 * `reuseExistingServer` (on locally) makes Playwright bind to ANY process that
 * answers on the port, and a generic `/health` check passes for foreign apps
 * too — that's how a stray dev server silently hijacked the run and produced 30+
 * cryptic "could not obtain a token" failures. `POST /api/auth/login` with junk
 * credentials is AutoModeler-specific: it returns 401 (bad creds) or 422
 * (validation). Anything else — 404/405 from a foreign route, 5xx from another
 * app, or a refused connection — means the port is NOT our backend. Allowlist
 * the expected statuses and fail loud, once, with an actionable message.
 */
const EXPECTED = new Set([401, 422])

export default async function globalSetup(): Promise<void> {
  const relocate =
    "Stop it, or relocate E2E to free ports: " +
    "E2E_BACKEND_URL=http://localhost:8100 E2E_BASE_URL=http://localhost:3100"
  let status: number
  try {
    const res = await fetch(`${BACKEND}/api/auth/login`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ email: "e2e-preflight@local", password: "x" }),
    })
    status = res.status
  } catch (err) {
    throw new Error(
      `E2E preflight: no server reachable at ${BACKEND} ` +
        `(${(err as Error).message}). Start the backend, or relocate with ` +
        `E2E_BACKEND_URL=http://localhost:8100 E2E_BASE_URL=http://localhost:3100`,
    )
  }
  if (!EXPECTED.has(status)) {
    throw new Error(
      `E2E preflight: the server on ${BACKEND} is NOT AutoModeler — ` +
        `POST /api/auth/login returned ${status}, expected 401/422. Another ` +
        `dev process is probably holding the port. ${relocate}`,
    )
  }
}
