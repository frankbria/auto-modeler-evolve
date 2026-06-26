import { BACKEND } from "./env"

/**
 * Preflight: confirm the server answering on BACKEND is actually AutoModeler.
 *
 * `reuseExistingServer` (on locally) makes Playwright bind to ANY process that
 * answers on the port, and a generic `/health` check passes for foreign apps
 * too — that's how a stray dev server silently hijacked the run and produced 30+
 * cryptic "could not obtain a token" failures. `POST /api/auth/login` is
 * AutoModeler-specific: it returns 401/422 for junk credentials, but 404 (or
 * refuses to connect) when the port holds something else or nothing. Fail loud
 * here, once, with an actionable message.
 */
export default async function globalSetup(): Promise<void> {
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
  if (status === 404) {
    throw new Error(
      `E2E preflight: a server is listening on ${BACKEND} but it is NOT ` +
        `AutoModeler (POST /api/auth/login -> 404). Another dev process is ` +
        `probably holding the port. Stop it, or relocate E2E to free ports: ` +
        `E2E_BACKEND_URL=http://localhost:8100 E2E_BASE_URL=http://localhost:3100`,
    )
  }
}
