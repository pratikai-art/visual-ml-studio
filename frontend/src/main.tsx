import React, { Component, type ErrorInfo, type ReactNode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./index.css";

type Props = { children: ReactNode };
type State = { error: Error | null };

class AppErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Visual ML Studio failed to render.", error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <main className="p-8">
          <h1 className="text-lg font-bold">Visual ML Studio could not render</h1>
          <p className="mt-2">{this.state.error.message}</p>
          <p className="text-mute mt-2">Reload the page. If the problem continues, start the backend and frontend using the commands in README.md.</p>
        </main>
      );
    }
    return this.props.children;
  }
}

const root = document.getElementById("root");
if (!root) throw new Error("Visual ML Studio root element is missing.");
createRoot(root).render(<AppErrorBoundary><React.StrictMode><App /></React.StrictMode></AppErrorBoundary>);
