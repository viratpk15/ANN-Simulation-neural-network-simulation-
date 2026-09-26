import { Component, type ErrorInfo, type ReactNode } from 'react'

interface Props { children: ReactNode }
interface State { error: Error | null }

/**
 * Last-resort guard: if any component throws during render, show a readable
 * error card instead of an unmounted (blank) page.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error('NeuroSim Lab crashed:', error, info.componentStack)
  }

  render() {
    if (this.state.error) {
      return (
        <div className="h-screen w-screen flex items-center justify-center bg-slate-950 text-slate-200 p-6">
          <div className="max-w-md text-center space-y-4">
            <div className="text-5xl">💥</div>
            <h1 className="text-xl font-semibold">Something went wrong</h1>
            <p className="text-sm text-slate-400 font-mono break-all">{this.state.error.message}</p>
            <button
              onClick={() => window.location.reload()}
              className="px-4 py-2 rounded bg-indigo-500 hover:bg-indigo-400 text-white text-sm font-medium"
            >
              Reload the app
            </button>
          </div>
        </div>
      )
    }
    return this.props.children
  }
}