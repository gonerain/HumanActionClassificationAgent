export const api = {
  async get(path: string) {
    const res = await fetch(path)
    if (!res.ok) throw new Error(`${res.status}`)
    return res.json()
  },
  async post(path: string, body: any) {
    const res = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
    if (!res.ok) throw new Error(`${res.status}`)
    return res.json()
  },
  async del(path: string) {
    const res = await fetch(path, { method: 'DELETE' })
    if (!res.ok) throw new Error(`${res.status}`)
    return res.json()
  },
}

