export function renderErrorPage() {
  return `<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Something went wrong</title>
    <style>
      body {
        margin: 0;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: #0f172a;
        color: #e2e8f0;
        display: flex;
        min-height: 100vh;
        align-items: center;
        justify-content: center;
      }
      main {
        max-width: 520px;
        padding: 2.5rem;
        text-align: center;
        background: rgba(15, 23, 42, 0.9);
        border: 1px solid rgba(148, 163, 184, 0.25);
        border-radius: 16px;
      }
      h1 {
        margin: 0 0 1rem;
        font-size: 2rem;
      }
      p {
        margin: 0 0 1.5rem;
        line-height: 1.6;
        color: #cbd5f5;
      }
      a {
        display: inline-block;
        padding: 0.6rem 1.2rem;
        border-radius: 999px;
        background: #6366f1;
        color: white;
        text-decoration: none;
        font-weight: 600;
      }
    </style>
  </head>
  <body>
    <main>
      <h1>We hit a snag</h1>
      <p>The app ran into an unexpected error. Please refresh the page or try again shortly.</p>
      <a href="/">Return home</a>
    </main>
  </body>
</html>`;
}
