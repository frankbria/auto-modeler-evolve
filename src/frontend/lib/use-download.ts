"use client"

import { useCallback, useState } from "react"
import { downloadFile } from "./api"

/**
 * React state wrapper around the header-authenticated `downloadFile` helper
 * (issue #28). Owner-scoped download buttons use this so they share one
 * implementation of the "preparing / error" UX while keeping the bearer token
 * in the request header (never the URL).
 *
 * Returns `download(url, filename?)` plus `downloading`/`error` state for the
 * button to reflect. `errorMessage` is shown if the request fails.
 */
export function useDownload(
  errorMessage = "Download failed. Please try again."
) {
  const [downloading, setDownloading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const download = useCallback(
    async (url: string, filename?: string) => {
      setDownloading(true)
      setError(null)
      try {
        await downloadFile(url, filename)
      } catch {
        setError(errorMessage)
      } finally {
        setDownloading(false)
      }
    },
    [errorMessage]
  )

  return { download, downloading, error }
}
