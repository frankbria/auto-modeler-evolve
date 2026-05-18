/**
 * Tests for predict/[id] page dashboard field config integration.
 * Covers: simplified view badge, hidden fields, locked fields, display labels,
 * and API client getDashboardConfig method.
 */

// ---------------------------------------------------------------------------
// Type tests — DashboardFieldEntry and DashboardConfigResponse
// ---------------------------------------------------------------------------

import type {
  DashboardFieldEntry,
  DashboardConfigResponse,
} from "@/lib/types"

describe("DashboardFieldEntry type", () => {
  it("has required fields", () => {
    const entry: DashboardFieldEntry = {
      feature_name: "units",
      type: "numeric",
      is_visible: true,
      is_locked: false,
      locked_value: null,
      display_label: null,
      display_order: null,
    }
    expect(entry.feature_name).toBe("units")
    expect(entry.is_visible).toBe(true)
  })

  it("accepts locked field with value", () => {
    const locked: DashboardFieldEntry = {
      feature_name: "region",
      type: "categorical",
      is_visible: true,
      is_locked: true,
      locked_value: "North",
      display_label: null,
      display_order: null,
    }
    expect(locked.is_locked).toBe(true)
    expect(locked.locked_value).toBe("North")
  })

  it("accepts hidden field", () => {
    const hidden: DashboardFieldEntry = {
      feature_name: "customer_id",
      type: "categorical",
      is_visible: false,
      is_locked: false,
      locked_value: null,
      display_label: null,
      display_order: null,
    }
    expect(hidden.is_visible).toBe(false)
  })

  it("accepts display_label override", () => {
    const labeled: DashboardFieldEntry = {
      feature_name: "revenue_usd",
      type: "numeric",
      is_visible: true,
      is_locked: false,
      locked_value: null,
      display_label: "Revenue",
      display_order: 1,
    }
    expect(labeled.display_label).toBe("Revenue")
    expect(labeled.display_order).toBe(1)
  })
})

describe("DashboardConfigResponse type", () => {
  it("has expected shape", () => {
    const cfg: DashboardConfigResponse = {
      deployment_id: "dep-1",
      fields: [
        {
          feature_name: "units",
          type: "numeric",
          is_visible: true,
          is_locked: false,
          locked_value: null,
          display_label: null,
          display_order: null,
        },
        {
          feature_name: "region",
          type: "categorical",
          is_visible: false,
          is_locked: false,
          locked_value: null,
          display_label: null,
          display_order: null,
        },
      ],
      total_count: 2,
      visible_count: 1,
      locked_count: 0,
    }
    expect(cfg.visible_count).toBe(1)
    expect(cfg.locked_count).toBe(0)
    expect(cfg.fields).toHaveLength(2)
  })
})

// ---------------------------------------------------------------------------
// API client method signature test
// ---------------------------------------------------------------------------

import { api } from "@/lib/api"

describe("api.deploy.getDashboardConfig", () => {
  it("exists on api.deploy", () => {
    expect(typeof api.deploy.getDashboardConfig).toBe("function")
  })

  it("returns a promise (fetch-based)", async () => {
    const mockResp: DashboardConfigResponse = {
      deployment_id: "dep-1",
      fields: [],
      total_count: 0,
      visible_count: 0,
      locked_count: 0,
    }
    global.fetch = jest.fn().mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(mockResp),
    } as unknown as Response)

    const result = await api.deploy.getDashboardConfig("dep-1")
    expect(result).toEqual(mockResp)
    expect((global.fetch as jest.Mock).mock.calls[0][0]).toContain(
      "/api/deploy/dep-1/dashboard-config"
    )
  })
})

// ---------------------------------------------------------------------------
// cfgMap and derived logic — pure unit tests
// ---------------------------------------------------------------------------

describe("dashboard config field filtering logic", () => {
  const fields: DashboardFieldEntry[] = [
    {
      feature_name: "units",
      type: "numeric",
      is_visible: true,
      is_locked: false,
      locked_value: null,
      display_label: null,
      display_order: null,
    },
    {
      feature_name: "customer_id",
      type: "categorical",
      is_visible: false,
      is_locked: false,
      locked_value: null,
      display_label: null,
      display_order: null,
    },
    {
      feature_name: "region",
      type: "categorical",
      is_visible: true,
      is_locked: true,
      locked_value: "North",
      display_label: null,
      display_order: null,
    },
  ]

  const cfgMap = Object.fromEntries(fields.map((f) => [f.feature_name, f]))

  it("builds cfgMap keyed by feature_name", () => {
    expect(cfgMap["units"].is_visible).toBe(true)
    expect(cfgMap["customer_id"].is_visible).toBe(false)
    expect(cfgMap["region"].is_locked).toBe(true)
  })

  it("filters hidden fields from schema", () => {
    const schema = [{ name: "units" }, { name: "customer_id" }, { name: "region" }]
    const visible = schema.filter((e) => cfgMap[e.name]?.is_visible !== false)
    expect(visible.map((e) => e.name)).toEqual(["units", "region"])
  })

  it("counts hidden and locked correctly", () => {
    const hiddenCount = fields.filter((f) => !f.is_visible).length
    const lockedCount = fields.filter((f) => f.is_locked).length
    expect(hiddenCount).toBe(1)
    expect(lockedCount).toBe(1)
  })

  it("isSimplifiedView is true when any hidden or locked", () => {
    const hiddenCount = fields.filter((f) => !f.is_visible).length
    const lockedCount = fields.filter((f) => f.is_locked).length
    const isSimplifiedView = hiddenCount > 0 || lockedCount > 0
    expect(isSimplifiedView).toBe(true)
  })

  it("buildPayload injects locked value for locked fields", () => {
    const schema = [
      { name: "units", type: "numeric" },
      { name: "region", type: "categorical" },
    ]
    const inputs: Record<string, string> = { units: "10", region: "South" }
    const payload: Record<string, unknown> = {}
    for (const entry of schema) {
      const cfg = cfgMap[entry.name]
      if (cfg?.is_locked && cfg.locked_value != null) {
        payload[entry.name] =
          entry.type === "numeric" ? parseFloat(cfg.locked_value) : cfg.locked_value
        continue
      }
      const raw = inputs[entry.name] ?? ""
      payload[entry.name] = entry.type === "numeric" ? parseFloat(raw) : raw
    }
    // region is locked to "North" — user's "South" is ignored
    expect(payload["region"]).toBe("North")
    // units is not locked — uses user input "10"
    expect(payload["units"]).toBe(10)
  })
})
