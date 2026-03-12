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

export default function HomePage() {
  const router = useRouter()
  const { projects, setProjects } = useAppStore()
  const [showForm, setShowForm] = useState(false)
  const [newName, setNewName] = useState("")
  const [creating, setCreating] = useState(false)
  const [loading, setLoading] = useState(true)

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

  return (
    <div className="mx-auto max-w-3xl px-6 py-12">
      <div className="mb-8">
        <h1 className="text-3xl font-bold tracking-tight">AutoModeler</h1>
        <p className="mt-1 text-muted-foreground">
          AI-powered data modeling
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
        <p className="text-sm text-muted-foreground">
          No projects yet. Create your first project above.
        </p>
      ) : (
        <div className="grid gap-3">
          {projects.map((project) => (
            <Card
              key={project.id}
              className="cursor-pointer transition-colors hover:bg-muted/50"
              onClick={() => router.push(`/project/${project.id}`)}
            >
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  {project.name}
                  <Badge variant={statusVariant(project.status)}>
                    {project.status}
                  </Badge>
                </CardTitle>
                {project.description && (
                  <CardDescription>{project.description}</CardDescription>
                )}
              </CardHeader>
              <CardContent>
                <p className="text-xs text-muted-foreground">
                  Created {formatDate(project.created_at)}
                </p>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
