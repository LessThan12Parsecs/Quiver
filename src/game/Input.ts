/**
 * Player input → SurfInput (docs/DESIGN.md "Controls").
 *
 *   key        prone                    standing / popping
 *   W / ↑      paddle                   weight forward (trim)
 *   S / ↓      —                        weight back
 *   A D / ← →  steer (paddle one side)  lean to the left / right rail
 *   Q / E      —                        twist (yaw torque)
 *   Shift      —                        crouch
 *   Space      pop up                   —
 * Game actions (callbacks): R reset at the lineup, T spawn on a wave, C camera, P pause,
 * [ / ] slower / faster, H help, G GUI, F force arrows, . step one frame, , step one physics step.
 *
 * Axes ramp toward their key targets (≈0.12 s) so a tap gives a partial lean. Edges (pop-up) are
 * latched until a physics step consumes them (`consumeEdges`), so they are never lost between
 * frames. Automation: `setKeys({ W: true })` holds virtual keys; `override` forces SurfInput
 * fields. Gamepad (standard mapping): left stick lean/steer + trim/paddle, RT paddle, LT crouch,
 * LB/RB or right stick twist, A pop up, X reset, Y wave, View/Back camera, Start pause.
 */
import { createSurfInput, type SurfInput } from '../physics/SurfSim';
import type { Stance } from '../physics/Rider';

export type GameAction =
  | 'reset'
  | 'spawnWave'
  | 'camera'
  | 'pause'
  | 'slower'
  | 'faster'
  | 'help'
  | 'gui'
  | 'forces'
  | 'stepFrame'
  | 'stepPhysics';

const ACTION_KEYS: Record<string, GameAction> = {
  KeyR: 'reset',
  KeyT: 'spawnWave',
  KeyC: 'camera',
  KeyP: 'pause',
  BracketLeft: 'slower',
  BracketRight: 'faster',
  KeyH: 'help',
  KeyG: 'gui',
  KeyF: 'forces',
  Period: 'stepFrame',
  Comma: 'stepPhysics',
};

/** Keys whose default browser action (scrolling) is suppressed. */
const BLOCK_DEFAULT = new Set(['Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight']);

/** Normalise automation key names: "W" → "KeyW", "Shift" → "ShiftLeft", "Up" → "ArrowUp". */
export function normalizeKey(k: string): string {
  if (/^[a-zA-Z]$/.test(k)) return `Key${k.toUpperCase()}`;
  if (/^[0-9]$/.test(k)) return `Digit${k}`;
  const lower = k.toLowerCase();
  if (lower === 'shift') return 'ShiftLeft';
  if (lower === 'space' || k === ' ') return 'Space';
  if (['up', 'down', 'left', 'right'].includes(lower)) return `Arrow${lower[0].toUpperCase()}${lower.slice(1)}`;
  return k;
}

function approach(x: number, target: number, rate: number, dt: number): number {
  const step = rate * dt;
  return x < target ? Math.min(x + step, target) : Math.max(x - step, target);
}

export class Input {
  /** SurfInput built by `update` (shared object: SurfSim reads it every step). */
  readonly surf: SurfInput = createSurfInput();
  /** Automation: forces these SurfInput fields (applied after keys/gamepad). */
  override: Partial<SurfInput> | null = null;
  /** Called for game-action keys/buttons. */
  onAction: ((a: GameAction, shift: boolean) => void) | null = null;
  /** True while a gamepad is connected and was used. */
  gamepadActive = false;

  private readonly down = new Set<string>();
  private readonly virtual = new Set<string>();
  private popUpPending = false;
  private side = 0;
  private fwd = 0;
  private twist = 0;
  private crouch = 0;
  private readonly padPrev: boolean[] = [];
  private readonly target: Window;

  private readonly onKeyDown = (e: KeyboardEvent): void => {
    if (isEditable(e.target)) return;
    if (BLOCK_DEFAULT.has(e.code)) e.preventDefault();
    if (e.repeat) return;
    this.down.add(e.code);
    if (e.code === 'Space') this.popUpPending = true;
    const action = ACTION_KEYS[e.code];
    if (action && !e.ctrlKey && !e.metaKey && !e.altKey) this.onAction?.(action, e.shiftKey);
  };
  private readonly onKeyUp = (e: KeyboardEvent): void => {
    this.down.delete(e.code);
  };
  private readonly onBlur = (): void => {
    this.down.clear();
  };

  constructor(target: Window = window) {
    this.target = target;
    target.addEventListener('keydown', this.onKeyDown);
    target.addEventListener('keyup', this.onKeyUp);
    target.addEventListener('blur', this.onBlur);
  }

  dispose(): void {
    this.target.removeEventListener('keydown', this.onKeyDown);
    this.target.removeEventListener('keyup', this.onKeyUp);
    this.target.removeEventListener('blur', this.onBlur);
  }

  /** 1 if the key is held, else 0. */
  private readonly key = (code: string): number => (this.isDown(code) ? 1 : 0);

  /** Is a key (KeyboardEvent.code) held, physically or virtually? */
  isDown(code: string): boolean {
    return this.down.has(code) || this.virtual.has(code);
  }

  /**
   * Automation: hold/release virtual keys, e.g. `{ W: true, Shift: true }` or `{ KeyD: false }`.
   * Pressing "Space" latches a pop-up; action keys (R, T, C, …) fire their action.
   */
  setKeys(keys: Record<string, boolean>): void {
    for (const [k, v] of Object.entries(keys)) {
      const code = normalizeKey(k);
      if (v) {
        if (!this.virtual.has(code)) {
          if (code === 'Space') this.popUpPending = true;
          const action = ACTION_KEYS[code];
          if (action) this.onAction?.(action, false);
        }
        this.virtual.add(code);
      } else {
        this.virtual.delete(code);
      }
    }
  }

  /** Release all virtual keys and clear the override. */
  clearAutomation(): void {
    this.virtual.clear();
    this.override = null;
  }

  /** Latch a pop-up edge (consumed by the next physics step). */
  requestPopUp(): void {
    this.popUpPending = true;
  }

  /**
   * Build `surf` for this frame. `dt` = real seconds since the last call (axis ramps; pass
   * Infinity to jump to the targets), `stance` selects the prone or standing mapping.
   */
  update(dt: number, stance: Stance): SurfInput {
    const k = this.key;
    const up = Math.max(k('KeyW'), k('ArrowUp'));
    const downK = Math.max(k('KeyS'), k('ArrowDown'));
    const left = Math.max(k('KeyA'), k('ArrowLeft'));
    const right = Math.max(k('KeyD'), k('ArrowRight'));
    let sideT = right - left;
    let fwdT = up - downK;
    let twistT = k('KeyE') - k('KeyQ');
    let crouchT = Math.max(k('ShiftLeft'), k('ShiftRight'));
    let paddle = up;

    // gamepad (first connected, standard mapping)
    const pad = this.readGamepad();
    if (pad) {
      const dz = (v: number): number => (Math.abs(v) < 0.15 ? 0 : (v - Math.sign(v) * 0.15) / 0.85);
      const lx = dz(pad.axes[0] ?? 0);
      const ly = dz(pad.axes[1] ?? 0);
      const rx = dz(pad.axes[2] ?? 0);
      const btn = (i: number): number => pad.buttons[i]?.value ?? 0;
      const any = Math.abs(lx) + Math.abs(ly) + Math.abs(rx) + btn(6) + btn(7) + btn(4) + btn(5) > 0.05;
      if (any) this.gamepadActive = true;
      if (Math.abs(lx) > Math.abs(sideT)) sideT = lx;
      if (Math.abs(ly) > Math.abs(fwdT)) fwdT = -ly;
      const tw = btn(5) - btn(4) + rx;
      if (Math.abs(tw) > Math.abs(twistT)) twistT = Math.max(-1, Math.min(1, tw));
      crouchT = Math.max(crouchT, btn(6));
      paddle = Math.max(paddle, btn(7), Math.max(-ly, 0));
      this.padButtons(pad);
    }

    const prone = stance === 'prone';
    const ramp = Number.isFinite(dt) ? dt : 1e3;
    this.side = approach(this.side, sideT, 8, ramp);
    this.fwd = approach(this.fwd, fwdT, 5, ramp);
    this.twist = approach(this.twist, twistT, 8, ramp);
    this.crouch = approach(this.crouch, crouchT, 5, ramp);
    const s = this.surf;
    s.paddle = prone ? paddle : 0;
    s.steer = prone ? this.side : 0;
    s.leanForward = prone ? 0 : this.fwd;
    s.leanSide = prone ? 0 : this.side;
    s.crouch = prone ? 0 : this.crouch;
    s.twist = prone ? 0 : this.twist;
    s.popUp = this.popUpPending;
    s.reset = false;
    if (this.override) Object.assign(s, this.override);
    return s;
  }

  /** Clear edge-triggered inputs after a physics step used them. */
  consumeEdges(): void {
    this.popUpPending = false;
    this.surf.popUp = false;
    this.surf.reset = false;
    if (this.override) {
      this.override.popUp = false;
      this.override.reset = false;
    }
  }

  private readGamepad(): Gamepad | null {
    const nav = this.target.navigator;
    if (!nav || typeof nav.getGamepads !== 'function') return null;
    const pads = nav.getGamepads();
    for (const p of pads) if (p && p.connected) return p;
    return null;
  }

  /** Edge-triggered gamepad buttons → pop-up and game actions. */
  private padButtons(pad: Gamepad): void {
    const map: Array<[number, GameAction | 'popUp']> = [
      [0, 'popUp'],
      [2, 'reset'],
      [3, 'spawnWave'],
      [8, 'camera'],
      [9, 'pause'],
    ];
    for (const [i, a] of map) {
      const pressed = pad.buttons[i]?.pressed ?? false;
      if (pressed && !this.padPrev[i]) {
        this.gamepadActive = true;
        if (a === 'popUp') this.popUpPending = true;
        else this.onAction?.(a, false);
      }
      this.padPrev[i] = pressed;
    }
  }
}

function isEditable(t: EventTarget | null): boolean {
  if (!t || typeof (t as HTMLElement).tagName !== 'string') return false;
  const tag = (t as HTMLElement).tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || (t as HTMLElement).isContentEditable;
}
