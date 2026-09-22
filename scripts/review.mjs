import { readFileSync, writeFileSync } from 'node:fs';
import { digestFiles } from './publication.mjs';

const slug = process.argv[2];
if (!slug || !process.argv.includes('--verified')) throw Error('Usage: pnpm review <slug> --verified (only after editorial review and experiment verification)');
const plan = JSON.parse(readFileSync('publication.json'));
const entry = plan.articles.find(item => item.slug === slug);
if (!entry) throw Error('Unknown article');
entry.reviewedHash = digestFiles(process.cwd(), entry.files);
entry.reviewedAt = new Date().toISOString();
writeFileSync('publication.json', JSON.stringify(plan, null, 2) + '\n');
console.log(`Recorded review for ${slug}.`);
