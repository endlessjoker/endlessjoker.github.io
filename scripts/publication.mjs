import { createHash } from 'node:crypto';
import { readFileSync, readdirSync } from 'node:fs';
import path from 'node:path';

export const sha256 = text => createHash('sha256').update(text).digest('hex');

export function digestFiles(root, files) {
  const values = [...files].sort().map(file => `${file}\0${readFileSync(path.join(root, file), 'utf8')}`);
  return sha256(values.join('\0'));
}

export function walk(dir) {
  return readdirSync(dir, { withFileTypes: true }).flatMap(entry => {
    const file = path.join(dir, entry.name);
    return entry.isDirectory() ? walk(file) : [file];
  });
}

export function validatePlan(plan) {
  const ids = new Set(), dates = new Set();
  for (const item of plan.articles) {
    if (!/^[a-z0-9-]+$/.test(item.slug) || ids.has(item.slug)) throw Error('Invalid or duplicate article slug');
    ids.add(item.slug);
    // Preserve the already-published launch timestamp; all new chapters release at 21:00.
    const initialLaunch = item.number === 1 && item.slug === 'reader-start' && item.publishAt === '2026-09-22T09:00:00+08:00';
    if ((!/^\d{4}-\d{2}-\d{2}T21:00:00\+08:00$/.test(item.publishAt) && !initialLaunch) || !Number.isFinite(Date.parse(item.publishAt))) {
      throw Error(`Invalid Beijing release date: ${item.slug}`);
    }
    const day = item.publishAt.slice(0, 10);
    if (dates.has(day)) throw Error('Only one article can be scheduled per Beijing day');
    dates.add(day);
    if (!item.files.every(f => /^(content|examples)\//.test(f) && !f.includes('..'))) throw Error('Public files must stay inside the blog');
    if (!item.files.includes(item.article)) throw Error('The article must be included in its review hash');
  }
}

export function releasedArticles(plan, now = new Date()) {
  validatePlan(plan);
  const sorted = [...plan.articles].sort((a, b) => Date.parse(a.publishAt) - Date.parse(b.publishAt));
  const released = [];
  for (const item of sorted) {
    // An unreviewed gap stops later articles too: this is a reading sequence.
    if (!item.reviewedHash || Date.parse(item.publishAt) > now.getTime()) break;
    released.push(item);
  }
  return released;
}

export function verifyReviewedFiles(plan, root) {
  for (const item of plan.articles) {
    if (item.reviewedHash && digestFiles(root, item.files) !== item.reviewedHash) {
      throw Error(`Reviewed content changed: ${item.slug}. Review the edits and experiment again before publication.`);
    }
  }
}
