import { readFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { validatePlan, verifyReviewedFiles, walk } from './publication.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const plan = JSON.parse(readFileSync(path.join(root, 'publication.json')));
validatePlan(plan);
verifyReviewedFiles(plan, root);
const forbidden = [
  /\/Users\/[^\s/]+\//,
  /\b(?:192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)\b/,
  /-----BEGIN [A-Z ]*PRIVATE KEY-----/,
  /\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}/,
  /\b[A-Za-z0-9._%+-]+@(?:qq|163|126)\.com\b/
];
for (const directory of ['content', 'examples']) {
  if (directory === 'examples' && !existsSync(path.join(root, directory))) continue;
  for (const file of walk(path.join(root, directory))) {
    const source = readFileSync(file, 'utf8');
    if (forbidden.some(pattern => pattern.test(source))) throw Error(`Publication boundary check failed: ${path.relative(root, file)}`);
    if (file.endsWith('.md') && (source.match(/^```/gm) || []).length % 2) throw Error(`Unclosed code fence: ${file}`);
  }
}
console.log(`Checked ${plan.articles.length} article records, review hashes and public-content boundaries.`);
