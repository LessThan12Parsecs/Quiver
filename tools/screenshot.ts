/**
 * Headless screenshots (Chromium + SwiftShader WebGL2) of the ocean viewer or any page of the app.
 *
 *   npm run shots                                    # default preset/time list
 *   npm run shots -- --presets beach,side --times 60,67.4 --q high
 *   npm run shots -- --shots beach@67.4,horizon@100   # explicit preset@time list
 *   npm run shots -- --url "/index.html?cam=follow" --name game_follow
 *   npm run shots -- --game "spawn=wave&cam=follow&paused=1" --eval "window.__quiver.stepSeconds(1)"
 *
 * Options:
 *   --presets a,b     viewer camera presets (lineup, beach, side, face, aerial, horizon)
 *   --times t1,t2     ocean times (s); every preset is shot at every time
 *   --shots p@t,...   explicit list instead of presets x times. A shot may also be a custom
 *                     camera "name=x,y,z>tx,ty,tz@t" (position > look-at); separate such shots
 *                     with ';', e.g. --shots "lip=-30,2,10>-34,0.5,0@67.4;beach@60"
 *   --q <quality>     ocean quality (low|medium|high|ultra|number)
 *   --extra "k=v&.."  extra query parameters for the viewer (e.g. "debug=1&sun=30,200")
 *   --url <path>      capture an arbitrary page path (repeatable); disables preset mode
 *   --game "<query>"  shorthand for --url "/index.html?<query>" (the game; repeatable)
 *   --name <file>     output base name for --url/--game shots (repeatable, matched by order)
 *   --eval "<js>"     expression evaluated (awaited) after the page is ready, before the shot;
 *                     the tool then waits for ready again (e.g. "window.__quiver.stepSeconds(2)").
 *                     Applies to the preceding --url/--game; before any of them, to all pages
 *   --ready <expr>    JS expression that becomes truthy when the page is ready
 *                     (default: window.__oceanView?.ready || window.__quiver?.ready)
 *   --wait <ms>       extra settle time after ready (default 0)
 *   --out <dir>       output directory (default tools/out)
 *   --width/--height  viewport (default 1280x720)
 *   --timeout <s>     per-shot timeout (default 300)
 *   --port <n>        Vite port (default 5287)
 * Prints one line per PNG plus page stats; exits non-zero on page errors or timeouts.
 */
import { mkdirSync } from 'node:fs';
import { resolve } from 'node:path';
import { chromium, type Page } from 'playwright';
import { createServer } from 'vite';

interface Shot {
  cam: string;
  t: number;
  pos?: [number, number, number];
  look?: [number, number, number];
}

function parseShot(spec: string): Shot {
  const at = spec.lastIndexOf('@');
  const head = at >= 0 ? spec.slice(0, at) : spec;
  const t = at >= 0 ? Number(spec.slice(at + 1)) : 67.4;
  if (!head.includes('>')) return { cam: head, t };
  const eq = head.indexOf('=');
  const name = eq >= 0 ? head.slice(0, eq) : 'custom';
  const [p, l] = head.slice(eq + 1).split('>');
  const vec = (v: string): [number, number, number] => {
    const a = v.split(',').map(Number);
    if (a.length !== 3 || !a.every(Number.isFinite)) throw new Error(`bad vector "${v}" in shot "${spec}"`);
    return a as [number, number, number];
  };
  return { cam: name, t, pos: vec(p), look: vec(l) };
}

interface Args {
  presets: string[];
  times: number[];
  shots: Shot[];
  q: string;
  extra: string;
  urls: string[];
  names: string[];
  /** Per-url eval (index-aligned with urls) and the default for urls without one. */
  evals: Array<string | undefined>;
  evalAll: string | undefined;
  ready: string;
  wait: number;
  out: string;
  width: number;
  height: number;
  timeout: number;
  port: number;
}

function parseArgs(argv: string[]): Args {
  const a: Args = {
    presets: ['beach', 'side', 'face', 'lineup', 'aerial', 'horizon'],
    times: [67.4],
    shots: [],
    q: 'high',
    extra: '',
    urls: [],
    names: [],
    evals: [],
    evalAll: undefined,
    ready: '(window.__oceanView && window.__oceanView.ready) || (window.__quiver && window.__quiver.ready)',
    wait: 0,
    out: 'tools/out',
    width: 1280,
    height: 720,
    timeout: 300,
    port: 5287,
  };
  for (let i = 0; i < argv.length; i++) {
    const k = argv[i];
    const v = argv[i + 1];
    const take = (): string => {
      i++;
      if (v === undefined) throw new Error(`missing value for ${k}`);
      return v;
    };
    switch (k) {
      case '--presets': a.presets = take().split(',').filter(Boolean); break;
      case '--times': a.times = take().split(',').map(Number); break;
      case '--shots':
      {
        const v = take();
        a.shots = (v.includes('>') || v.includes(';') ? v.split(';') : v.split(',')).filter(Boolean).map(parseShot);
        break;
      }
      case '--q': a.q = take(); break;
      case '--extra': a.extra = take(); break;
      case '--url': a.urls.push(take()); break;
      case '--game': a.urls.push(`/index.html?${take().replace(/^\?/, '')}`); break;
      case '--eval':
        if (a.urls.length) a.evals[a.urls.length - 1] = take();
        else a.evalAll = take();
        break;
      case '--name': a.names.push(take()); break;
      case '--ready': a.ready = take(); break;
      case '--wait': a.wait = Number(take()); break;
      case '--out': a.out = take(); break;
      case '--width': a.width = Number(take()); break;
      case '--height': a.height = Number(take()); break;
      case '--timeout': a.timeout = Number(take()); break;
      case '--port': a.port = Number(take()); break;
      default: throw new Error(`unknown option ${k}`);
    }
  }
  if (!a.shots.length) for (const cam of a.presets) for (const t of a.times) a.shots.push({ cam, t });
  return a;
}

const args = parseArgs(process.argv.slice(2));
const outDir = resolve(args.out);
mkdirSync(outDir, { recursive: true });

const server = await createServer({ server: { port: args.port, strictPort: false }, logLevel: 'error' });
await server.listen();
const base = server.resolvedUrls!.local[0].replace(/\/$/, '');
const browser = await chromium.launch({
  args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
});

let failed = false;
const errors: string[] = [];

async function waitReady(page: Page, label: string): Promise<void> {
  try {
    await page.waitForFunction(args.ready, null, { timeout: args.timeout * 1000, polling: 250 });
  } catch {
    throw new Error(`timeout waiting for "${args.ready}" (${label})`);
  }
  if (args.wait > 0) await page.waitForTimeout(args.wait);
}

async function stats(page: Page): Promise<string> {
  const s = await page.evaluate(() => {
    const w = window as unknown as { __oceanView?: { stats?: Record<string, unknown> }; __quiver?: { stats?: Record<string, unknown> } };
    return w.__oceanView?.stats ?? w.__quiver?.stats ?? null;
  });
  if (!s) return '';
  const keys = ['patches', 'vertices', 'triangles', 'levels', 'beachVertices', 'drawCalls', 'time', 'stance', 'speed', 'camera', 'oceanVertices'];
  return Object.entries(s)
    .filter(([k]) => keys.includes(k))
    .map(([k, v]) => `${k}=${v}`)
    .join(' ');
}

try {
  const page = await browser.newPage({ viewport: { width: args.width, height: args.height }, deviceScaleFactor: 1 });
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(`[console.error] ${m.text()}`);
  });
  page.on('pageerror', (e) => errors.push(`[pageerror] ${e.message}`));

  if (args.urls.length) {
    for (let i = 0; i < args.urls.length; i++) {
      const u = args.urls[i];
      const name = args.names[i] ?? u.replace(/^\/+/, '').replace(/[^a-zA-Z0-9._-]+/g, '_');
      const t0 = Date.now();
      await page.goto(base + (u.startsWith('/') ? u : `/${u}`));
      await waitReady(page, u);
      const ev = args.evals[i] ?? args.evalAll;
      if (ev) {
        await page.evaluate(`(async () => { await (${ev}); })()`);
        await page.evaluate('window.__quiver && window.__quiver.render ? window.__quiver.render() : null');
        await waitReady(page, `${u} after eval`);
      }
      const file = resolve(outDir, `${name}.png`);
      await page.screenshot({ path: file, timeout: args.timeout * 1000 });
      console.log(`${file}  (${((Date.now() - t0) / 1000).toFixed(1)} s) ${await stats(page)}`);
    }
  } else {
    let loaded = false;
    for (const shot of args.shots) {
      const t0 = Date.now();
      const label = `${shot.cam}@${shot.t}`;
      if (!loaded) {
        const q = new URLSearchParams({ cam: shot.cam, t: String(shot.t), q: args.q, clean: '1' });
        if (shot.pos) q.set('pos', shot.pos.join(','));
        if (shot.look) q.set('look', shot.look.join(','));
        const extra = args.extra ? `&${args.extra}` : '';
        await page.goto(`${base}/tools/ocean-view/index.html?${q.toString()}${extra}`);
        await waitReady(page, label);
        loaded = true;
      } else {
        await page.evaluate(
          (v) => (window as unknown as { __oceanView: { setView(v: unknown): Promise<void> } }).__oceanView.setView(v),
          { cam: shot.pos ? undefined : shot.cam, t: shot.t, pos: shot.pos, look: shot.look },
        );
        await waitReady(page, label);
      }
      const file = resolve(outDir, `${shot.cam}_${shot.t}.png`);
      await page.screenshot({ path: file, timeout: args.timeout * 1000 });
      console.log(`${file}  (${((Date.now() - t0) / 1000).toFixed(1)} s) ${await stats(page)}`);
    }
  }
} catch (e) {
  failed = true;
  console.error(String((e as Error)?.message ?? e));
} finally {
  await browser.close();
  await server.close();
}
const real = errors.filter((e) => !/GL Driver Message|GPU stall/.test(e));
if (real.length) {
  failed = true;
  console.error(real.join('\n'));
}
process.exit(failed ? 1 : 0);
