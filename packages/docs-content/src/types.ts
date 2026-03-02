export type DocFrontmatter = {
  title?: string;
  description?: string;
  order?: number;
  category?: string;
  collapsed?: boolean;
};

export type DocMetadata = {
  title: string;
  description: string;
  order: number;
  category: string;
  collapsed?: boolean;
};

export type DocRecord = {
  slug: string;
  segments: string[];
  metadata: DocMetadata;
  raw: string;
};

export type DocsIndex = {
  docs: DocRecord[];
};
