import test from 'node:test';
import assert from 'node:assert/strict';
import { releasedArticles, validatePlan, verifyReviewedFiles, digestFiles } from './publication.mjs';
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import path from 'node:path';

const article = (slug, date, reviewedHash = 'reviewed') => ({ slug, publishAt: `${date}T21:00:00+08:00`, reviewedHash, article: `content/${slug}.md`, files: [`content/${slug}.md`] });
const plan = { articles: [article('one', '2026-09-22'), article('two', '2026-09-23')] };
test('Beijing 21:00 boundary is independent of the machine timezone', () => {
  assert.equal(releasedArticles(plan, new Date('2026-09-23T12:59:59Z')).length, 1);
  assert.equal(releasedArticles(plan, new Date('2026-09-23T13:00:00Z')).length, 2);
  assert.equal(releasedArticles(plan, new Date('2026-09-21T23:00:00Z')).length, 0);
});
test('reruns do not advance the sequence; unreviewed gaps stop it', () => {
  const blocked = { articles: [article('one', '2026-09-22'), article('two', '2026-09-23', ''), article('three', '2026-09-24')] };
  const now = new Date('2026-10-01T13:00:00Z');
  assert.deepEqual(releasedArticles(blocked, now).map(x => x.slug), ['one']);
  assert.deepEqual(releasedArticles(blocked, now), releasedArticles(blocked, now));
});
test('cannot schedule two articles on the same day or import outside files', () => {
  assert.throws(() => validatePlan({ articles: [article('one', '2026-09-22'), article('two', '2026-09-22')] }));
  assert.throws(() => validatePlan({ articles: [{ ...article('one', '2026-09-22'), files: ['../private.swift'] }] }));
});
test('editing the article or its experiment invalidates review', () => {
  const root = mkdtempSync(path.join(tmpdir(), 'public-blog-policy-'));
  try {
    mkdirSync(path.join(root, 'content')); mkdirSync(path.join(root, 'examples'));
    writeFileSync(path.join(root, 'content/one.md'), '# One');
    writeFileSync(path.join(root, 'examples/check.swift'), 'print(1)');
    const entry = { ...article('one', '2026-09-22'), files: ['content/one.md', 'examples/check.swift'] };
    entry.reviewedHash = digestFiles(root, entry.files);
    verifyReviewedFiles({ articles: [entry] }, root);
    writeFileSync(path.join(root, 'examples/check.swift'), 'print(2)');
    assert.throws(() => verifyReviewedFiles({ articles: [entry] }, root));
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('only the original launch may retain a morning timestamp', () => {
  const launch = { ...article('reader-start', '2026-09-22'), number: 1, publishAt: '2026-09-22T09:00:00+08:00' };
  assert.doesNotThrow(() => validatePlan({ articles: [launch] }));
  assert.throws(() => validatePlan({ articles: [{ ...article('two', '2026-09-23'), publishAt: '2026-09-23T09:00:00+08:00' }] }));
  assert.throws(() => validatePlan({ articles: [{ ...launch, publishAt: '2026-09-23T09:00:00+08:00' }] }));
});
