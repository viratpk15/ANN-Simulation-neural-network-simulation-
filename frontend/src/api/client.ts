// Thin typed REST client. Same-origin: vite proxies /api in dev,
// the backend serves the SPA in production.

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method,
    headers: body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })
  if (!res.ok) {
    let msg = `${res.status} ${res.statusText}`
    try {
      const data = await res.json()
      if (typeof data.detail === 'string') msg = data.detail
      else if (Array.isArray(data.errors)) msg = data.errors.map((e: { msg?: string }) => e.msg).join('; ')
    } catch { /* keep status message */ }
    throw new ApiError(res.status, msg)
  }
  return res.json() as Promise<T>
}

export const api = {
  get: <T>(path: string) => request<T>('GET', path),
  post: <T>(path: string, body?: unknown) => request<T>('POST', path, body),
  del: <T>(path: string) => request<T>('DELETE', path),

  async upload<T>(path: string, file: File): Promise<T> {
    const form = new FormData()
    form.append('file', file)
    const res = await fetch(path, { method: 'POST', body: form })
    if (!res.ok) {
      let msg = `${res.status} ${res.statusText}`
      try {
        const data = await res.json()
        if (typeof data.detail === 'string') msg = data.detail
      } catch { /* ignore */ }
      throw new ApiError(res.status, msg)
    }
    return res.json() as Promise<T>
  },
}
