/**
 * Tests for the header-authenticated `downloadFile` helper (issue #28).
 *
 * Proves the security-critical contract:
 *   - the request carries `Authorization: Bearer <token>` (header auth)
 *   - the bearer token never appears in the URL/query string
 *   - the response is downloaded as a Blob via an object URL
 *   - the filename comes from Content-Disposition, falling back otherwise
 *   - a failed request throws so callers can show an error
 */

import { downloadFile } from "../lib/api"
import { setToken, clearToken } from "../lib/auth-token"

const TOKEN = "header.jwt.secret"

// --- DOM / object-URL plumbing -------------------------------------------
const mockCreateObjectURL = jest.fn().mockReturnValue("blob:fake-url")
const mockRevokeObjectURL = jest.fn()
Object.defineProperty(URL, "createObjectURL", {
  value: mockCreateObjectURL,
  configurable: true,
})
Object.defineProperty(URL, "revokeObjectURL", {
  value: mockRevokeObjectURL,
  configurable: true,
})

const originalCreateElement = document.createElement.bind(document)
let mockAnchor: HTMLAnchorElement | null = null
jest.spyOn(document, "createElement").mockImplementation((tag: string) => {
  if (tag === "a") {
    mockAnchor = originalCreateElement("a") as HTMLAnchorElement
    jest.spyOn(mockAnchor, "click").mockImplementation(() => {})
    return mockAnchor
  }
  return originalCreateElement(tag)
})

const mockBlob = new Blob(["file body"], { type: "text/csv" })

function mockFetch(
  ok: boolean,
  disposition: string | null = null,
  status = ok ? 200 : 500
) {
  const fn = jest.fn().mockResolvedValue({
    ok,
    status,
    blob: () => Promise.resolve(mockBlob),
    headers: {
      get: (name: string) =>
        name.toLowerCase() === "content-disposition" ? disposition : null,
    },
  } as unknown as Response)
  global.fetch = fn
  return fn
}

beforeEach(() => {
  jest.clearAllMocks()
  mockCreateObjectURL.mockReturnValue("blob:fake-url")
  setToken(TOKEN)
})

afterEach(() => {
  clearToken()
})

describe("downloadFile", () => {
  it("sends the bearer token in the Authorization header (not the URL)", async () => {
    const fetchMock = mockFetch(true, 'attachment; filename="report.csv"')

    await downloadFile("/api/models/run-1/report")

    const [calledUrl, init] = fetchMock.mock.calls[0]
    const headers = new Headers((init as RequestInit).headers)
    expect(headers.get("Authorization")).toBe(`Bearer ${TOKEN}`)
    // The token must NEVER leak into the URL / query string.
    expect(String(calledUrl)).not.toContain(TOKEN)
    expect(String(calledUrl)).not.toMatch(/access_token|[?&]token=/)
  })

  it("resolves a relative backend path against API_URL", async () => {
    const fetchMock = mockFetch(true)
    await downloadFile("/api/data/ds-1/download", "data.csv")
    const [calledUrl] = fetchMock.mock.calls[0]
    expect(String(calledUrl)).toMatch(/^https?:\/\/.+\/api\/data\/ds-1\/download$/)
  })

  it("passes an absolute URL through unchanged", async () => {
    const fetchMock = mockFetch(true)
    const abs = "http://localhost:8000/api/deploy/dep-1/export"
    await downloadFile(abs)
    expect(String(fetchMock.mock.calls[0][0])).toBe(abs)
  })

  it("triggers a blob download via an object URL", async () => {
    mockFetch(true, 'attachment; filename="report.csv"')
    await downloadFile("/api/models/run-1/report")
    expect(mockCreateObjectURL).toHaveBeenCalledWith(mockBlob)
    expect(mockAnchor?.getAttribute("download")).toBe("report.csv")
    expect(mockAnchor?.click).toHaveBeenCalled()
    expect(mockRevokeObjectURL).toHaveBeenCalledWith("blob:fake-url")
  })

  it("uses the fallback filename when no Content-Disposition is present", async () => {
    mockFetch(true, null)
    await downloadFile("/api/data/ds-1/download", "fallback.csv")
    expect(mockAnchor?.getAttribute("download")).toBe("fallback.csv")
  })

  it("keeps a literal '%' in a plain filename without throwing (no decode)", async () => {
    // A plain `filename=` value is NOT percent-encoded; decoding it would throw
    // URIError and abort the download after the blob was already fetched.
    mockFetch(true, 'attachment; filename="sales 50% off.csv"')
    await downloadFile("/api/data/ds-1/download", "fallback.csv")
    expect(mockAnchor?.getAttribute("download")).toBe("sales 50% off.csv")
  })

  it("decodes an RFC 5987 extended filename* value", async () => {
    mockFetch(true, "attachment; filename*=UTF-8''report%20q1.csv")
    await downloadFile("/api/models/run-1/report", "fallback.csv")
    expect(mockAnchor?.getAttribute("download")).toBe("report q1.csv")
  })

  it("refuses a cross-origin absolute URL without sending the token", async () => {
    const fetchMock = mockFetch(true)
    await expect(
      downloadFile("https://evil.example.com/api/models/run-1/report")
    ).rejects.toThrow(/non-API origin/)
    expect(fetchMock).not.toHaveBeenCalled()
    expect(mockCreateObjectURL).not.toHaveBeenCalled()
  })

  it("throws when the request fails", async () => {
    mockFetch(false, null, 403)
    await expect(downloadFile("/api/models/run-1/report")).rejects.toThrow(
      /Download failed/
    )
    expect(mockCreateObjectURL).not.toHaveBeenCalled()
  })
})
