/**
 * DOM HUD (markup and styles live in index.html): speed, stance, board, sunk volume, wave state
 * under the board, attitude, a balance dial (ankle torque / what the feet can hold, roll → x,
 * pitch → y; the dashed ring is the slip limit), a badge (CAUGHT while prone and the board runs
 * with the wave — the moment to pop up; RIDING + timer while standing on a wave), ride timer,
 * wipeout messages (the advice follows how the rider fell), hints and a status line. Text
 * refreshes at ~10 Hz, the balance dot every frame. Rides are the game's (standing) rides, see
 * Game.stepOnce — not `telemetry.riding`, which also counts prone belly rides and lone boards.
 */
import type { SurfTelemetry } from '../physics/SurfSim';
import type { Rider, WipeoutReason } from '../physics/Rider';

export interface HudInfo {
  boardName: string;
  boardVolumeL: number;
  fps: number;
  timeScale: number;
  paused: boolean;
  camera: string;
  /** Standing on a wave right now, and for how long (s). */
  riding: boolean;
  rideTime: number;
  /** Prone and the board runs with the wave (pop-up cue). */
  caught: boolean;
  /** Rides: last finished and best this session, s. */
  lastRide: number;
  bestRide: number;
  /** Which way the body went over in a balance wipeout. */
  tipOver: Rider['tipOver'];
  gamepad: boolean;
  /** Real-time factor actually achieved (sim seconds per real second / time scale). */
  simLoad: number;
  /** Autopilot phase when the scripted surfer is in control, else null. */
  autopilot: string | null;
}

const WIPEOUT_TEXT: Record<WipeoutReason, [string, string]> = {
  balance: ['Lost your balance', 'the board rolled further than your feet could hold — ease off the rail (A/D) and keep the dot in the ring'],
  flipped: ['Board flipped', 'rolled past the rail'],
  pearl: ['Pearled', 'the nose dug in — weight back on steep drops'],
  buried: ['Buried', 'the board went too deep'],
  impact: ['Slammed', 'the legs could not absorb the impact'],
};

/** Balance wipeouts by the direction the body went over (the generic text above is 'side'). */
const BALANCE_TEXT: Record<'forward' | 'back', [string, string]> = {
  back: [
    'Fell off the back',
    'the board stalled under you (the wave passed or the whitewater stopped it) — pop up only once it runs with the wave (CAUGHT), weight forward (W)',
  ],
  forward: ['Thrown over the nose', 'the board dropped away under you — weight back (S) on steep drops'],
};

function el<T extends HTMLElement = HTMLElement>(id: string): T {
  const e = document.getElementById(id);
  if (!e) throw new Error(`HUD element #${id} missing from index.html`);
  return e as T;
}

export class Hud {
  private readonly speed = el('hud-speed');
  private readonly speedMs = el('hud-speed-ms');
  private readonly stance = el('hud-stance');
  private readonly board = el('hud-board');
  private readonly volume = el('hud-volume');
  private readonly volumeBar = el('hud-volume-bar');
  private readonly wave = el('hud-wave');
  private readonly fullness = el('hud-fullness');
  private readonly fullnessBar = el('hud-fullness-bar');
  private readonly water = el('hud-water');
  private readonly attitude = el('hud-attitude');
  private readonly rideTime = el('hud-ridetime');
  private readonly balancePanel = el('hud-balance');
  private readonly balanceDot = el('balance-dot');
  private readonly balanceValue = el('balance-value');
  private readonly balanceDetail = el('balance-detail');
  private readonly ride = el('hud-ride');
  private readonly message = el('hud-message');
  private readonly messageBig = el('hud-message-big');
  private readonly messageSmall = el('hud-message-small');
  private readonly hint = el('hud-hint');
  private readonly status = el('hud-status');
  private readonly help = el('help');

  private lastText = -1;
  private lastStance = '';
  private flashUntil = 0;
  /** The message shown is the wipeout message (cleared when the rider is back up). */
  private showingWipeout = false;
  private hintText = '';
  private badge = '';

  /** Show/hide the controls panel. */
  get helpVisible(): boolean {
    return !this.help.classList.contains('hidden');
  }

  set helpVisible(v: boolean) {
    this.help.classList.toggle('hidden', !v);
  }

  /** Big centred message for `seconds` (real time). */
  flash(big: string, small: string, seconds: number, now: number): void {
    this.messageBig.textContent = big;
    this.messageSmall.textContent = small;
    this.message.classList.add('on');
    this.message.classList.remove('bad');
    this.flashUntil = now + seconds;
    this.showingWipeout = false;
  }

  /** Bottom hint line (null hides it). */
  setHint(text: string | null): void {
    const t = text ?? '';
    if (t === this.hintText) return;
    this.hintText = t;
    this.hint.textContent = t;
    this.hint.classList.toggle('on', t.length > 0);
  }

  /** `now` = real time in seconds (performance.now() / 1000). */
  update(t: SurfTelemetry, info: HudInfo, now: number): void {
    const standing = t.stance === 'standing' || t.stance === 'popping';
    // balance dial every frame
    const bx = clampMag(t.balanceRoll, t.balancePitch, 1.6);
    const r = 23; // ring radius in px = balance 1
    this.balanceDot.style.transform = standing ? `translate(${(bx[0] * r).toFixed(1)}px, ${(-bx[1] * r).toFixed(1)}px)` : 'none';
    const b = standing ? t.balance : 0;
    const col = b < 0.7 ? 'var(--accent)' : b < 1 ? 'var(--warn)' : 'var(--bad)';
    this.balanceDot.style.background = col;
    this.balanceDot.style.boxShadow = `0 0 ${t.balanceTimer > 0 ? 14 : 8}px ${col}`;

    // messages: wipeouts persist until the stance changes; flashes time out
    if (t.stance !== this.lastStance) {
      if (t.stance === 'fallen' && t.wipeoutReason) {
        const tip = info.tipOver;
        const [big, small] =
          t.wipeoutReason === 'balance' && (tip === 'back' || tip === 'forward') ? BALANCE_TEXT[tip] : WIPEOUT_TEXT[t.wipeoutReason];
        this.messageBig.textContent = `Wipeout — ${big}`;
        this.messageSmall.textContent = `${small}.\nR: back to the lineup · T: next wave`;
        this.message.classList.add('on', 'bad');
        this.flashUntil = Infinity;
        this.showingWipeout = true;
      } else if (this.lastStance === 'fallen' && this.showingWipeout) {
        // (a flash shown since, e.g. T's "Drop in!", keeps its own timeout)
        this.message.classList.remove('on', 'bad');
        this.flashUntil = 0;
        this.showingWipeout = false;
      }
      this.lastStance = t.stance;
      this.lastText = -1;
    }
    if (this.flashUntil !== Infinity && this.flashUntil > 0 && now > this.flashUntil) {
      this.message.classList.remove('on');
      this.flashUntil = 0;
    }
    const badge = info.riding ? 'riding' : info.caught ? 'caught' : '';
    if (badge !== this.badge) {
      this.badge = badge;
      this.ride.classList.toggle('on', badge !== '');
      this.ride.classList.toggle('caught', badge === 'caught');
      if (badge === 'caught') this.ride.textContent = 'CAUGHT — POP UP';
      this.lastText = -1;
    }

    if (now - this.lastText < 0.1) return;
    this.lastText = now;
    this.speed.textContent = (t.speed * 3.6).toFixed(1);
    this.speedMs.textContent = `${t.speed.toFixed(2)} m/s`;
    this.stance.textContent = t.stance;
    this.board.textContent = `${info.boardName} ${info.boardVolumeL.toFixed(0)} L`;
    const sunk = info.boardVolumeL > 0 ? t.submergedLiters / info.boardVolumeL : 0;
    this.volume.textContent = `${t.submergedLiters.toFixed(1)} L · body ${t.riderSubmergedLiters.toFixed(0)} L`;
    setBar(this.volumeBar, sunk, sunk > 0.95 ? 'var(--bad)' : sunk > 0.75 ? 'var(--warn)' : 'var(--accent)');
    const w = t.water;
    this.wave.textContent = `${w.waveHeight.toFixed(2)} m · ${waveState(t)}`;
    this.fullness.textContent = `fullness ${w.fullness.toFixed(2)}${w.breaking > 0.02 ? ` · white ${w.breaking.toFixed(2)}` : ''}`;
    setBar(this.fullnessBar, w.fullness, w.breaking > 0.2 ? '#ffffff' : w.fullness > 0.8 ? 'var(--warn)' : 'var(--accent)');
    this.water.textContent = `${w.depth.toFixed(1)} m · deck ${deckText(t.deckDepth)}`;
    this.attitude.textContent = `${deg(t.pitchRad)} pitch ${deg(t.rollRad)} roll${t.finStalled ? ' · fin STALL' : ''}`;
    this.rideTime.textContent = info.riding
      ? `${info.rideTime.toFixed(1)} s`
      : info.bestRide > 0
        ? `last ${info.lastRide.toFixed(1)} s · best ${info.bestRide.toFixed(1)} s`
        : '—';
    if (badge === 'riding') this.ride.textContent = `RIDING  ${info.rideTime.toFixed(1)} s`;
    this.balancePanel.classList.toggle('off', !standing);
    this.balanceValue.textContent = standing ? `${(t.balance * 100).toFixed(0)} %` : '—';
    this.balanceDetail.textContent = standing
      ? `feet ${(t.feetLoadN / 9.81).toFixed(0)} kg${t.balanceTimer > 0 ? ` · slip ${(t.balanceTimer * 1000).toFixed(0)} ms` : ''}`
      : t.stance === 'prone'
        ? `paddle ${t.paddleThrustN.toFixed(0)} N`
        : '';
    const sim = info.simLoad < 0.97 && !info.paused ? ` · sim ${(info.simLoad * 100).toFixed(0)}%` : '';
    this.status.innerHTML =
      `${info.autopilot ? `<b>AUTOPILOT</b> (${info.autopilot}) · ` : ''}` +
      `${info.paused ? '<b>PAUSED</b> · ' : ''}t ${t.time.toFixed(1)} s · ` +
      `${info.timeScale === 1 ? '×1' : `<b>×${fmtScale(info.timeScale)}</b>`} · ${info.camera} cam · ` +
      `${info.fps.toFixed(0)} fps${sim}${info.gamepad ? ' · gamepad' : ''}`;
  }
}

function waveState(t: SurfTelemetry): string {
  const w = t.water;
  if (w.waveHeight < 0.25) return 'flat';
  if (w.breaking > 0.3) return 'whitewater';
  const p = w.wavePhase;
  let where = p > -0.35 && p < 0.35 ? 'crest' : p >= 0.35 && p < 2.6 ? 'face' : p <= -0.35 && p > -2.6 ? 'back' : 'trough';
  if (where === 'face' && w.fullness > 0.8) where = 'steep face';
  if (w.ratio > 1 && w.breaking < 0.3) where += ' (reformed)';
  return where;
}

/** Deck height relative to the water: "+3 cm" above, "−7 cm" awash/below. */
function deckText(depth: number): string {
  const cm = Math.round(-depth * 100);
  return `${cm > 0 ? '+' : cm < 0 ? '−' : ''}${Math.abs(cm)} cm`;
}

function deg(r: number): string {
  const d = (r * 180) / Math.PI;
  return `${d >= 0 ? '+' : ''}${d.toFixed(0)}°`;
}

function fmtScale(s: number): string {
  return s >= 1 ? s.toFixed(0) : s >= 0.25 ? s.toFixed(2).replace(/0$/, '') : `1/${Math.round(1 / s)}`;
}

function setBar(e: HTMLElement, f: number, color: string): void {
  e.style.width = `${(Math.min(Math.max(f, 0), 1) * 100).toFixed(1)}%`;
  e.style.background = color;
}

const _clamp: [number, number] = [0, 0];
function clampMag(x: number, y: number, m: number): [number, number] {
  const l = Math.hypot(x, y);
  const k = l > m ? m / l : 1;
  _clamp[0] = Number.isFinite(x) ? x * k : 0;
  _clamp[1] = Number.isFinite(y) ? y * k : 0;
  return _clamp;
}
