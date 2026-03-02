import { promises as fs } from "node:fs";
import path from "node:path";
import matter from "gray-matter";

type DocMetadata = {
  title: string;
  description: string;
  order: number;
  category: string;
  collapsed?: boolean;
};

type DocRecord = {
  slug: string;
  segments: string[];
  metadata: DocMetadata;
  raw: string;
};

const REPO_ROOT = path.resolve(import.meta.dir, "..", "..", "..");
const DOCS_ROOT = path.join(REPO_ROOT, "docs");
const OUTPUT_DIR = path.resolve(import.meta.dir, "..", "src", "generated");
const OUTPUT_FILE = path.join(OUTPUT_DIR, "docs-data.ts");

function startCase(input: string): string {
  return input
    .replace(/[-_]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

async function listMarkdownFiles(dir: string): Promise<string[]> {
  const entries = await fs.readdir(dir, { withFileTypes: true });
  const nested = await Promise.all(
    entries.map(async (entry) => {
      const absolutePath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        return listMarkdownFiles(absolutePath);
      }

      if (entry.isFile() && entry.name.endsWith(".md")) {
        return [absolutePath];
      }

      return [];
    })
  );

  return nested.flat();
}

function normalizeOrder(value: unknown): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }

  return 100;
}

async function createDocsData(): Promise<DocRecord[]> {
  const files = await listMarkdownFiles(DOCS_ROOT);
  const docs = await Promise.all(
    files.map(async (absolutePath) => {
      const relativePath = path.relative(DOCS_ROOT, absolutePath).replace(/\\/g, "/");
      const slug = relativePath.replace(/\.md$/i, "");
      const segments = slug.split("/");
      const source = await fs.readFile(absolutePath, "utf8");
      const parsed = matter(source);
      const data = parsed.data as Record<string, unknown>;

      const tail = segments.at(-1) ?? "overview";
      const head = segments.at(0) ?? "overview";

      const metadata: DocMetadata = {
        title:
          typeof data.title === "string" && data.title.trim().length > 0
            ? data.title
            : startCase(tail),
        description: typeof data.description === "string" ? data.description : "",
        order: normalizeOrder(data.order),
        category:
          typeof data.category === "string" && data.category.trim().length > 0
            ? data.category
            : startCase(head) || "Overview",
        ...(data.collapsed !== undefined ? { collapsed: Boolean(data.collapsed) } : {})
      };

      return {
        slug,
        segments,
        metadata,
        raw: parsed.content
      } satisfies DocRecord;
    })
  );

  docs.sort((a, b) => a.slug.localeCompare(b.slug));
  return docs;
}

async function main() {
  const docsData = await createDocsData();

  const output = `import type { DocRecord } from "../types.js";\n\nexport const docsData: DocRecord[] = ${JSON.stringify(docsData, null, 2)};\n`;

  await fs.mkdir(OUTPUT_DIR, { recursive: true });
  await fs.writeFile(OUTPUT_FILE, output, "utf8");

  process.stdout.write(`Generated ${docsData.length} docs records at ${OUTPUT_FILE}\n`);
}

main().catch((error) => {
  process.stderr.write(`${String(error)}\n`);
  process.exit(1);
});
