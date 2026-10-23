// Run an R script inside webR (R compiled to WebAssembly) with a host directory mounted.
//
// Usage (from this directory, after `npm ci --ignore-scripts`):
//   node run_r.mjs <script.R> <io-dir> [R packages...]
//
// The io-dir is mounted at /io inside R, so the script reads inputs from and writes outputs
// to /io/... . R packages are installed from the webR binary repository (repo.r-wasm.org)
// into the in-memory R library on every run; nothing is installed on the host.
//
// webR 0.6.0 on Windows passes a bare `C:\...` path to a dynamic `import()`, which Node's ESM
// loader rejects. patchWorker() rewrites that one call to use a file:// URL. It is idempotent
// and only touches the gitignored node_modules copy.
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const dist = path.join(here, 'node_modules', 'webr', 'dist');

function patchWorker() {
  const worker = path.join(dist, 'webr-worker.js');
  const src = fs.readFileSync(worker, 'utf8');
  const bad = 'await import((await import("path")).default.resolve(e))';
  const good =
    'await import((await import("url")).pathToFileURL((await import("path")).default.resolve(e)).href)';
  if (src.includes(bad)) fs.writeFileSync(worker, src.replace(bad, good));
}

const [script, ioDir, ...packages] = process.argv.slice(2);
if (!script || !ioDir) {
  console.error('usage: node run_r.mjs <script.R> <io-dir> [R packages...]');
  process.exit(2);
}
patchWorker();
const { WebR } = await import('webr');
const rel = path.relative(process.cwd(), dist).split(path.sep).join('/');
const webR = new WebR({ baseUrl: (rel.startsWith('.') ? rel : './' + rel) + '/' });
await webR.init();
await webR.FS.mkdir('/io');
await webR.FS.mount('NODEFS', { root: path.resolve(ioDir) }, '/io');
if (packages.length) await webR.installPackages(packages, { quiet: true });
const code = fs.readFileSync(script, 'utf8');
const shelter = await new webR.Shelter();
try {
  const out = await shelter.captureR(code, { withAutoprint: true });
  for (const line of out.output) {
    if (line.type === 'stdout') console.log(line.data);
    else console.error(line.data);
  }
} finally {
  shelter.purge();
}
process.exit(0);
