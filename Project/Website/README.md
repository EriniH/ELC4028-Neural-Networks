# Word Embeddings Tutorial

An interactive tutorial for exploring word embeddings, nearest neighbors, vector arithmetic, and related NLP concepts. The app includes a main English experience built with TanStack Start, plus a standalone Arabic embeddings demo.

## Features

- Interactive embedding visualization
- Nearest-neighbor search and similarity exploration
- Word analogy examples such as `king - man + woman ≈ queen`
- Short quiz to check understanding
- Standalone Arabic demo at `/arabic-embeddings-demo.html`
- Embedded Arabic demo inside the main page

## Tech Stack

- React 19
- TanStack Start and TanStack Router
- Vite
- TypeScript
- Tailwind CSS 4
- Radix UI primitives
- Lucide icons

## Getting Started

Install dependencies:

```bash
npm install
```

Run the development server:

```bash
npm run dev
```

Build the app for production:

```bash
npm run build
```

Preview the production build locally:

```bash
npm run preview
```

Run linting:

```bash
npm run lint
```

Format the codebase:

```bash
npm run format
```

## Project Structure

- `src/routes/index.tsx` - main landing page and Arabic demo embed
- `src/components/EmbeddingDemo.tsx` - interactive embedding demo
- `src/components/Quiz.tsx` - quiz UI
- `public/arabic-embeddings-demo.html` - standalone Arabic demo
- `src/lib/i18n.tsx` - language and translation helpers

## Notes

- The main page uses a language toggle for English and Arabic.
- The Arabic demo can be opened directly or viewed inside the embedded section on the homepage.
- The repository includes a `wrangler.jsonc`, so the app is likely intended to be deployable in Cloudflare-related environments.
