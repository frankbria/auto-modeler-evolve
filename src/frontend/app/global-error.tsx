"use client"

/**
 * Root error boundary (#23). Only fires when the root layout itself throws, so
 * it must render its own <html>/<body>. Kept deliberately dependency-free for
 * the same reason. Normal page errors are handled by app/error.tsx.
 */
export default function GlobalError({ reset }: { reset: () => void }) {
  return (
    <html lang="en">
      <body>
        <div
          role="alert"
          style={{
            minHeight: "60vh",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: "0.75rem",
            textAlign: "center",
            fontFamily: "sans-serif",
          }}
        >
          <p style={{ fontWeight: 600 }}>Something went wrong.</p>
          <p style={{ fontSize: "0.875rem", color: "#666" }}>
            The app hit an unexpected error.
          </p>
          <button
            onClick={reset}
            style={{
              padding: "0.5rem 1rem",
              border: "1px solid #ccc",
              borderRadius: "0.5rem",
              cursor: "pointer",
            }}
          >
            Try again
          </button>
        </div>
      </body>
    </html>
  )
}
