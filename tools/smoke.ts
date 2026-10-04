/**
 * End-to-end smoke test of the playable build (npm run smoke):
 * starts Vite, loads the game in headless Chromium (SwiftShader WebGL2) and checks that
 *   1. it boots without console errors, page errors or shader compile errors, renders a
 *      non-black frame and the physics state is finite;
 *   2. holding W at the lineup paddles the board forward (vs. a no-input control run from the
 *      same start time);
 *   3. Space pops the rider up (stance leaves 'prone');
 *   4. T spawns the rider standing on a wave moving faster than 2.5 m/s, still up after 0.5 s;
 *   5. R resets to the lineup (prone, x ≈ −75);
 *   6. live changes the debug GUI makes work: every board preset (mesh + physics swap), rider
 *      mass, an ocean/sandbar rebuild (beach mesh regenerated) — no errors, physics finite;
 *   7. the real-time loop (unpaused) advances the simulation;
 * and writes screenshots to tools/out/: smoke_lineup_paddle.png (follow cam, paddling),
 * smoke_wave_follow.png, smoke_wave_beach.png, smoke_wave_side.png. Exits 1 on any failure.
 *
 * Options: --port <n> (default 5293), --out <dir>, --width/--height (default 960x540),
 *          --q <ocean quality> (default low), --timeout <s per wait> (default 240),
 *          --dist (test the production build in dist/ via `vite preview`; run npm run build first).
 */
import { existsSync, mkdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { chromium, type Page } from 'playwright';
import { createServer, preview } from 'vite';
import type { PixelStats, QuiverApi, QuiverState } from '../src/game/Game';

function arg(name: string, def: string): string {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : def;
}

const port = Number(arg('port', '5293'));
const outDir = resolve(arg('out', 'tools/out'));
const width = Number(arg('width', '960'));
const height = Number(arg('height', '540'));
const quality = arg('q', 'low');
const timeoutMs = Number(arg('timeout', '240')) * 1000;
const useDist = process.argv.includes('--dist');
mkdirSync(outDir, { recursive: true });
if (useDist && !existsSync(resolve('dist/index.html'))) {
  console.error('dist/index.html not found: run `npm run build` first');
  process.exit(1);
}

type W = Window & { __quiver: QuiverApi };

const failures: string[] = [];
const notes: string[] = [];
function check(ok: boolean, what: string, detail = ''): void {
  const line = `${ok ? 'PASS' : 'FAIL'}  ${what}${detail ? `  (${detail})` : ''}`;
  console.log(line);
  if (!ok) failures.push(line);
}
const f2 = (v: number): string => (Number.isFinite(v) ? v.toFixed(2) : String(v));

let base: string;
let closeServer: () => Promise<void>;
if (useDist) {
  const pv = await preview({ preview: { port, strictPort: false }, logLevel: 'error' });
  base = pv.resolvedUrls!.local[0].replace(/\/$/, '');
  closeServer = () => pv.close();
  console.log(`      testing the production build (dist/) at ${base}`);
} else {
  const server = await createServer({ server: { port, strictPort: false }, logLevel: 'error' });
  await server.listen();
  base = server.resolvedUrls!.local[0].replace(/\/$/, '');
  closeServer = () => server.close();
}
const browser = await chromium.launch({
  args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
});

const consoleErrors: string[] = [];
const t0 = Date.now();

async function waitReady(page: Page, label: string): Promise<void> {
  try {
    await page.waitForFunction(() => {
      const q = (window as unknown as Partial<W>).__quiver;
      return !!q && q.ready;
    }, null, { timeout: timeoutMs, polling: 200 });
  } catch {
    throw new Error(`timeout waiting for the game to be ready (${label})`);
  }
}

async function state(page: Page): Promise<QuiverState> {
  return page.evaluate(() => (window as unknown as W).__quiver.state());
}

async function stepSeconds(page: Page, s: number): Promise<QuiverState> {
  return page.evaluate((sec) => (window as unknown as W).__quiver.stepSeconds(sec), s);
}

async function shot(page: Page, name: string): Promise<void> {
  await page.evaluate(() => (window as unknown as W).__quiver.render());
  await waitReady(page, name);
  const file = resolve(outDir, `${name}.png`);
  await page.screenshot({ path: file, timeout: timeoutMs });
  notes.push(file);
  console.log(`      wrote ${file}`);
}

function describe(s: QuiverState): string {
  return `t=${f2(s.time)} ${s.stance} pos=(${f2(s.x)}, ${f2(s.y)}, ${f2(s.z)}) v=${f2(s.speed)} m/s heading=${f2(s.headingDeg)}° sub=${f2(s.submergedLiters)} L`;
}

try {
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
  page.on('console', (m) => {
    const text = m.text();
    const bad = /Shader Error|WebGLProgram|GL_INVALID|INVALID_OPERATION|INVALID_VALUE|INVALID_ENUM|CONTEXT_LOST/i.test(text);
    if ((m.type() === 'error' && !/GPU stall|GL Driver Message/.test(text)) || bad) consoleErrors.push(`[console.${m.type()}] ${text}`);
  });
  page.on('pageerror', (e) => consoleErrors.push(`[pageerror] ${e.message}`));

  // ---------------------------------------------------------------- 1. boot
  const url = `${base}/index.html?paused=1&q=${quality}&spawn=lineup&help=0&gui=0&dpr=1`;
  await page.goto(url);
  await waitReady(page, 'boot');
  const s0 = await state(page);
  console.log(`      boot ${((Date.now() - t0) / 1000).toFixed(1)} s: ${describe(s0)}`);
  check(s0.finite, 'physics state finite after boot');
  check(s0.stance === 'prone', 'spawns prone at the lineup', s0.stance);
  check(Math.abs(s0.x + 75) < 0.5 && Math.abs(s0.z - 5) < 0.5, 'lineup position (−75, 5)', `${f2(s0.x)}, ${f2(s0.z)}`);
  const px: PixelStats = await page.evaluate(() => (window as unknown as W).__quiver.pixelStats());
  check(px.mean > 25 && px.mean < 235 && px.black < 0.05, 'renders a non-black, non-blown-out frame', `mean ${f2(px.mean)}, black ${f2(px.black * 100)}%, white ${f2(px.white * 100)}%`);

  // ---------------------------------------------------------------- 2. paddling
  const startT = s0.time;
  const h0 = (s0.headingDeg * Math.PI) / 180;
  const along = (a: QuiverState, b: QuiverState): number => (b.x - a.x) * Math.cos(h0) + (b.z - a.z) * Math.sin(h0);
  // control run: no input for 4 s
  const c1 = await stepSeconds(page, 4);
  const drift = along(s0, c1);
  // same start, W held (real key events through the game's Input)
  await page.evaluate((t) => (window as unknown as W).__quiver.game.resetLineup(t), startT);
  const p0 = await state(page);
  await page.keyboard.down('KeyW');
  const p1 = await stepSeconds(page, 4);
  const moved = along(p0, p1);
  console.log(`      paddle 4 s: ${describe(p1)}; moved ${f2(moved)} m vs drift ${f2(drift)} m`);
  check(p1.finite, 'physics state finite while paddling');
  check(p1.stance === 'prone', 'still prone while paddling', p1.stance);
  check(moved - drift > 2 && moved > 1.5, 'W paddles the board forward', `moved ${f2(moved)} m, control ${f2(drift)} m`);
  check(p1.speed > 0.6, 'paddling speed > 0.6 m/s', `${f2(p1.speed)} m/s`);
  await shot(page, 'smoke_lineup_paddle');
  await page.keyboard.up('KeyW');

  // ---------------------------------------------------------------- 3. pop-up
  await page.keyboard.press('Space');
  const u1 = await stepSeconds(page, 0.3);
  check(u1.stance !== 'prone', 'Space starts the pop-up', u1.stance);
  const u2 = await stepSeconds(page, 0.6);
  check(u2.finite, 'physics state finite after the pop-up', describe(u2));
  console.log(`      after pop-up: ${describe(u2)}`);

  // ---------------------------------------------------------------- 4. wave spawn
  await page.keyboard.press('KeyT');
  const w0 = await state(page);
  console.log(`      T: ${describe(w0)}  wave ${f2(w0.waveHeight)} m fullness ${f2(w0.fullness)}`);
  check(w0.stance === 'standing', 'T spawns the rider standing', w0.stance);
  check(w0.speed > 2.5, 'moving with the wave (> 2.5 m/s)', `${f2(w0.speed)} m/s`);
  check(w0.waveHeight > 1, 'on a set wave (> 1 m)', `${f2(w0.waveHeight)} m`);
  const w1 = await stepSeconds(page, 0.5);
  console.log(`      +0.5 s: ${describe(w1)} riding=${w1.riding} ${w1.ridingTime.toFixed(2)} s`);
  check(w1.finite, 'physics state finite on the wave');
  check(w1.stance === 'standing' && w1.speed > 2.5, 'still standing and > 2.5 m/s after 0.5 s', `${w1.stance}, ${f2(w1.speed)} m/s`);
  const deckOk = w1.y > -2 && w1.y < 3;
  check(deckOk, 'board near the water surface', `y=${f2(w1.y)}`);
  await shot(page, 'smoke_wave_follow');
  const pw: PixelStats = await page.evaluate(() => (window as unknown as W).__quiver.pixelStats());
  check(pw.mean > 25 && pw.black < 0.05, 'wave frame renders', `mean ${f2(pw.mean)}, black ${f2(pw.black * 100)}%`);
  await page.evaluate(() => (window as unknown as W).__quiver.setCamera('beach'));
  await shot(page, 'smoke_wave_beach');
  await page.evaluate(() => (window as unknown as W).__quiver.setCamera('side'));
  await shot(page, 'smoke_wave_side');
  await page.evaluate(() => (window as unknown as W).__quiver.setCamera('follow'));

  // ---------------------------------------------------------------- 5. reset
  await page.keyboard.press('KeyR');
  const r0 = await state(page);
  check(r0.stance === 'prone' && Math.abs(r0.x + 75) < 0.5, 'R resets to the lineup', describe(r0));

  // ---------------------------------------------------------------- 6. GUI-driven changes
  for (const id of ['shortboard', 'longboard', 'softtop', 'funboard']) {
    await page.evaluate((b) => (window as unknown as W).__quiver.game.setBoard(b as 'funboard'), id);
    await page.evaluate(() => (window as unknown as W).__quiver.render());
    const sb = await stepSeconds(page, 0.5);
    check(sb.board === id && sb.finite, `board swap → ${id}`, describe(sb));
  }
  await page.evaluate(() => (window as unknown as W).__quiver.game.setRiderMass(95));
  const sm = await stepSeconds(page, 0.5);
  check(sm.finite, 'rider mass 95 kg', describe(sm));
  await page.evaluate(() => {
    const g = (window as unknown as W).__quiver.game;
    const cfg = JSON.parse(JSON.stringify(g.ocean.config));
    cfg.swells[0].height = 1.4;
    cfg.bathymetry.bar.peakDepth = 1.2;
    g.applyOceanConfig(cfg, true);
  });
  await page.evaluate(() => (window as unknown as W).__quiver.render());
  const so = await stepSeconds(page, 0.5);
  check(so.finite, 'ocean + sandbar rebuild (beach mesh regenerated)', describe(so));

  // ---------------------------------------------------------------- 7. real-time loop
  const before = await state(page);
  await page.evaluate(() => (window as unknown as W).__quiver.setPaused(false));
  // SwiftShader frames take seconds: wait until the loop has run a few physics steps
  const t1 = Date.now();
  try {
    await page.waitForFunction((t) => (window as unknown as W).__quiver.state().time > t + 0.03, before.time, { timeout: timeoutMs, polling: 250 });
  } catch {
    /* reported by the check below */
  }
  console.log(`      real-time loop: ${((Date.now() - t1) / 1000).toFixed(1)} s wall clock`);
  await page.evaluate(() => (window as unknown as W).__quiver.setPaused(true));
  await waitReady(page, 'pause');
  const after = await state(page);
  check(after.time > before.time && after.finite, 'unpaused loop advances the simulation', `${f2(after.time - before.time)} s simulated`);

  const pageErrors: string[] = await page.evaluate(() => (window as unknown as W).__quiver.errors);
  consoleErrors.push(...pageErrors.map((e) => `[window.onerror] ${e}`));
} catch (e) {
  failures.push(`FAIL  ${String((e as Error)?.message ?? e)}`);
  console.error(String((e as Error)?.stack ?? e));
} finally {
  await browser.close();
  await closeServer();
}

check(consoleErrors.length === 0, 'no console / page / shader errors', consoleErrors.slice(0, 5).join(' | '));
console.log(`\nsmoke: ${failures.length ? `${failures.length} failure(s)` : 'all checks passed'} in ${((Date.now() - t0) / 1000).toFixed(0)} s`);
if (failures.length) console.log(failures.join('\n'));
process.exit(failures.length ? 1 : 0);
