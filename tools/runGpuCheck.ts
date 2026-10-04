/**
 * Starts a Vite dev server, opens tools/gpu-check in headless Chromium (WebGL2 via SwiftShader)
 * and fails if the GPU wave shader disagrees with the CPU wave model.
 *   npm run check:gpu
 */
import { chromium } from 'playwright';
import { createServer } from 'vite';

const server = await createServer({ server: { port: 5199, strictPort: false }, logLevel: 'error' });
await server.listen();
const url = `${server.resolvedUrls!.local[0]}tools/gpu-check/index.html`;
const browser = await chromium.launch({
  args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader', '--ignore-gpu-blocklist'],
});
let failed = true;
try {
  const page = await browser.newPage();
  const logs: string[] = [];
  page.on('console', (m) => logs.push(`[${m.type()}] ${m.text()}`));
  page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));
  await page.goto(url);
  await page.waitForFunction(() => (window as unknown as { __gpuCheck?: unknown }).__gpuCheck !== undefined, null, {
    timeout: 180_000,
  });
  const result = await page.evaluate(() => (window as unknown as { __gpuCheck: unknown }).__gpuCheck);
  console.log(JSON.stringify(result, null, 2));
  if (logs.length) console.log(logs.join('\n'));
  failed = !(result as { ok: boolean }).ok;
} finally {
  await browser.close();
  await server.close();
}
process.exit(failed ? 1 : 0);
