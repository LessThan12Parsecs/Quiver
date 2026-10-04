/**
 * Quiver test build bootstrap: checks WebGL2, creates the Game from the URL parameters and
 * exposes the automation hook on window.__quiver (see src/game/Game.ts).
 */
import { Game, parseGameParams, type QuiverApi } from './game/Game';

declare global {
  interface Window {
    __quiver?: QuiverApi;
  }
}

function fatal(msg: string): void {
  const el = document.getElementById('fatal');
  if (el) {
    el.textContent = msg;
    el.style.display = 'flex';
  }
  console.error(`[quiver] ${msg}`);
}

const probe = document.createElement('canvas').getContext('webgl2');
probe?.getExtension('WEBGL_lose_context')?.loseContext(); // only a capability check
if (!probe) {
  fatal('Quiver needs WebGL2, which this browser/GPU does not provide.');
} else {
  try {
    const game = new Game(document.getElementById('app')!, parseGameParams(new URLSearchParams(location.search)));
    window.__quiver = game.api;
    game.start();
  } catch (e) {
    fatal(`Failed to start: ${(e as Error)?.message ?? e}`);
    throw e;
  }
}
