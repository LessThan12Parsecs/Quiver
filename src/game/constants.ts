/** Game-shell constants shared by Game and the debug GUI. */

/** Physics steps per rendered frame at most (≈ 42 ms of simulated time at 240 Hz). */
export const MAX_SUBSTEPS = 10;
/** Selectable time scales ([ and ]). */
export const TIME_SCALES = [1 / 16, 1 / 8, 1 / 4, 1 / 2, 1, 2];
/** Default start time (s): the first set of the session is building at the peak. */
export const DEFAULT_START_TIME = 20;
/** Wave-spawn side of the peak: +1 (+Z), −1 (−Z), 0 = alternate. */
export type SpawnSide = 1 | -1 | 0;
