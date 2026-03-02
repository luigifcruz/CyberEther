# @cyberether/docs-content

`@cyberether/docs-content` exposes CyberEther documentation content and metadata as a plain TypeScript API.

This package does not include rendering/UI logic. It only provides normalized docs records.

## Local Development

This package is scaffolded as a private local package (`private: true`). It is consumed from the CyberEther monorepo/workspace, not installed from a public registry.

```bash
cd packages/docs-content
bun install
```

## Usage

```ts
import { getDocs, getDocBySlug } from "@cyberether/docs-content";

const docs = getDocs();
const overview = getDocBySlug("overview");
const install = getDocBySlug(["installation"]);
```

For local package checks/builds:

```bash
bun run check
bun run build
```

## Source of Truth

Markdown source files come from `CyberEther/docs/*.md` (including nested paths). The package generates `src/generated/docs-data.ts` automatically on `check` and `build` (not on plain install), and that generated file is intentionally not committed to git.
