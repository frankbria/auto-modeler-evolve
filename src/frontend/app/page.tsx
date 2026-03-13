"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardHeader,
  CardTitle,
  CardDescription,
  CardContent,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { api } from "@/lib/api"
import { useAppStore } from "@/lib/store"
import type { Project } from "@/lib/types"

export default function HomePage() {
  const router = useRouter()
  const { projects, setProjects } = useAppStore()
  const [showForm, setShowForm] = useState(false)
  const [newName, setNewName] = useState("")
  const [creating, setCreating] = useState(false)
  const [loading, setLoading] = useState(true)
  const [renamingId, setRenamingId] = useState<string | null>(null)
  const [renameValue, setRenameValue] = useState("")

  useEffect(() => {
    api.projects
      .list()
      .then(setProjects)
      .catch(() => setProjects([]))
      .finally(() => setLoading(false))
  }, [setProjects])

  async function handleCreate() {
    if (!newName.trim()) return
    setCreating(true)
    try {
      const project = await api.projects.create(newName.trim())
      setProjects([...projects, project])
      setNewName("")
      setShowForm(false)
    } finally {
      setCreating(false)
    }
  }

  async function handleDelete(id: string, e: React.MouseEvent) {
    e.stopPropagation()
    if (!confirm("Delete this project? This cannot be undone.")) return
    await api.projects.delete(id)
    setProjects(projects.filter((p) => p.id !== id))
  }

  async function handleDuplicate(id: string, e: React.MouseEvent) {
    e.stopPropagation()
    const copy = await api.projects.duplicate(id)
    setProjects([...projects, copy])
  }

  function startRename(project: Project, e: React.MouseEvent) {
    e.stopPropagation()
    setRenamingId(project.id)
    setRenameValue(project.name)
  }

  async function commitRename(id: string) {
    if (!renameValue.trim()) {
      setRenamingId(null)
      return
    }
    const updated = await api.projects.update(id, { name: renameValue.trim() })
    setProjects(projects.map((p) => (p.id === id ? { ...p, ...updated } : p)))
    setRenamingId(null)
  }

  function statusVariant(status: string) {
    switch (status) {
      case "deployed":
        return "default" as const
      case "modeling":
        return "secondary" as const
      default:
        return "outline" as const
    }
  }

  function formatDate(dateStr: string) {
    return new Date(dateStr).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
      year: "numeric",
    })
  }

  function formatRows(n: number) {
    if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M rows`
    if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K rows`
    return `${n} rows`
  }

  return (
    <div className="mx-auto max-w-3xl px-6 py-12">
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight">AutoModeler</h1>
        <p className="mt-1 text-muted-foreground">
          AI-powered data modeling — upload a spreadsheet, get predictions
        </p>
      </div>

      <div className="mb-6">
        {showForm ? (
          <div className="flex items-center gap-2">
            <Input
              placeholder="Project name"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") handleCreate()
                if (e.key === "Escape") {
                  setShowForm(false)
                  setNewName("")
                }
              }}
              autoFocus
            />
            <Button onClick={handleCreate} disabled={creating}>
              {creating ? "Creating..." : "Create"}
            </Button>
            <Button
              variant="ghost"
              onClick={() => {
                setShowForm(false)
                setNewName("")
              }}
            >
              Cancel
            </Button>
          </div>
        ) : (
          <Button onClick={() => setShowForm(true)}>New Project</Button>
        )}
      </div>

      {loading ? (
        <p className="text-sm text-muted-foreground">Loading projects...</p>
      ) : projects.length === 0 ? (
        <div className="rounded-xl border border-dashed p-10 text-center">
          <p className="text-sm font-medium">No projects yet</p>
          <p className="mt-1 text-xs text-muted-foreground">
            Create a project to get started. Upload a CSV and AutoModeler will
            guide you from data to a live prediction API — no code required.
          </p>
          <Button
            className="mt-4"
            variant="outline"
            onClick={() => setShowForm(true)}
          >
            Create your first project
          </Button>
        </div>
      ) : (
        <div className="grid gap-3">
          {projects.map((project) => (
            <Card
              key={project.id}
              data-testid="project-card"
              className="cursor-pointer transition-colors hover:bg-muted/50"
              onClick={() => router.push(`/project/${project.id}`)}
            >
              <CardHeader className="pb-2">
                <CardTitle className="flex items-center gap-2">
                  {renamingId === project.id ? (
                    <Input
                      value={renameValue}
                      onChange={(e) => setRenameValue(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") commitRename(project.id)
                        if (e.key === "Escape") setRenamingId(null)
                      }}
                      onBlur={() => commitRename(project.id)}
                      onClick={(e) => e.stopPropagation()}
                      autoFocus
                      className="h-7 text-sm font-semibold"
                    />
                  ) : (
                    <span className="truncate">{project.name}</span>
                  )}
                  <Badge variant={statusVariant(project.status)}>
                    {project.status}
                  </Badge>
                  {project.has_deployment && (
                    <Badge variant="default" className="text-xs">
                      live
                    </Badge>
                  )}
                </CardTitle>
                {project.description && (
                  <CardDescription>{project.description}</CardDescription>
                )}
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between">
                  <div className="flex flex-wrap gap-3 text-xs text-muted-foreground">
                    <span>Modified {formatDate(project.updated_at)}</span>
                    {project.dataset_filename && (
                      <span>
                        {project.dataset_filename}
                        {project.dataset_rows != null &&
                          ` · ${formatRows(project.dataset_rows)}`}
                      </span>
                    )}
                    {(project.model_count ?? 0) > 0 && (
                      <span>
                        {project.model_count} model
                        {project.model_count !== 1 ? "s" : ""} trained
                      </span>
                    )}
                  </div>
                  <div className="flex shrink-0 gap-1">
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-7 px-2 text-xs"
                      onClick={(e) => startRename(project, e)}
                    >
                      Rename
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-7 px-2 text-xs"
                      onClick={(e) => handleDuplicate(project.id, e)}
                    >
                      Duplicate
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-7 px-2 text-xs text-destructive hover:text-destructive"
                      onClick={(e) => handleDelete(project.id, e)}
                    >
                      Delete
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
