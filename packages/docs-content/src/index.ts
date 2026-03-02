import { docsData } from "./generated/docs-data.js";
import type { DocRecord, DocsIndex } from "./types.js";

function cloneDoc(doc: DocRecord): DocRecord {
  return {
    slug: doc.slug,
    segments: [...doc.segments],
    metadata: {
      title: doc.metadata.title,
      description: doc.metadata.description,
      order: doc.metadata.order,
      category: doc.metadata.category,
      ...(doc.metadata.collapsed !== undefined
        ? { collapsed: doc.metadata.collapsed }
        : {})
    },
    raw: doc.raw
  };
}

function normalizeSlug(input: string | string[]): string {
  if (Array.isArray(input)) {
    return input.filter(Boolean).join("/").replace(/^\/+|\/+$/g, "");
  }

  return input.replace(/^\/+|\/+$/g, "");
}

export function getDocs(): DocRecord[] {
  return docsData.map(cloneDoc);
}

export function getDocsIndex(): DocsIndex {
  return { docs: getDocs() };
}

export function getDocBySlug(slug: string | string[]): DocRecord | undefined {
  const normalizedSlug = normalizeSlug(slug);
  const doc = docsData.find((entry) => entry.slug === normalizedSlug);
  return doc ? cloneDoc(doc) : undefined;
}

export type { DocFrontmatter, DocMetadata, DocRecord, DocsIndex } from "./types.js";
