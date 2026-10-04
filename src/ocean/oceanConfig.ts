/**
 * Ocean configuration: swell, wind chop, breaking and surf-spot bathymetry.
 *
 * Coordinate convention (shared by the whole project):
 *   +Y up, mean sea level at y = 0.
 *   +X points toward the beach (swell travels roughly +X).
 *   Z runs along the shore.
 * Units: metres, seconds, kilograms, radians (configs use degrees where noted).
 */

/** Number of waves in a set pattern (the pattern repeats every SET_LENGTH waves). */
export const SET_LENGTH = 16;
/** Maximum number of shoaling swell components (GPU uniform array size). */
export const MAX_SWELLS = 3;
/** Maximum number of deep-water wind-chop components that affect geometry + physics. */
export const MAX_WIND_WAVES = 16;

export interface SwellConfig {
  /** Wave period in seconds. */
  period: number;
  /** Deep-water wave height (crest to trough) of a wave whose set-pattern value is 1, metres. */
  height: number;
  /** Deep-water propagation direction, degrees from +X toward +Z. */
  directionDeg: number;
  /** Phase offset in radians. */
  phase: number;
  /** Relative wave heights for consecutive waves, length SET_LENGTH. Shorter arrays are cycled. */
  setPattern: number[];
}

export interface WindWaveConfig {
  /** Number of chop components (<= MAX_WIND_WAVES). */
  count: number;
  /** Wind speed at 10 m (m/s); drives the Pierson-Moskowitz spectrum. */
  speed: number;
  /** Mean propagation direction of the chop, degrees from +X toward +Z. */
  directionDeg: number;
  /** Directional spread (half-angle), degrees. */
  spreadDeg: number;
  /** Shortest wavelength included in geometry/physics (shorter detail is shader-only normals). */
  minWavelength: number;
  /** Longest wavelength included. */
  maxWavelength: number;
  /** Multiplier on spectrum amplitudes. */
  amplitudeScale: number;
  /** Horizontal (Gerstner) displacement factor 0..1 before the steepness budget is applied. */
  choppiness: number;
  /** Maximum total horizontal-displacement steepness sum(k*Ah) the chop may use. */
  steepnessBudget: number;
  /** RNG seed for directions/phases. */
  seed: number;
}

export interface BarConfig {
  enabled: boolean;
  /** X position of the bar crest at the peak (z = 0). */
  x0: number;
  /** How far the bar crest recedes shoreward per metre of |z| (creates the A-frame peel). */
  sweep: number;
  /** Rounding radius for the |z| kink at the peak, metres. */
  roundness: number;
  /** Water depth over the bar crest at the peak, metres. */
  peakDepth: number;
  /** Increase of crest depth per metre of |z| (controls peel speed). */
  depthSlope: number;
  /** Gaussian half-width of the bar cross-section along x, metres. */
  width: number;
  /** |z| where the bar starts fading out. */
  fadeStartZ: number;
  /** |z| where the bar has fully faded out. */
  fadeEndZ: number;
}

export interface BathymetryConfig {
  /**
   * Base depth profile control points [x, depth] with depth positive below sea level
   * (negative depth = dry land). Interpolated with monotone cubic (PCHIP). Must be sorted by x.
   */
  profile: Array<[number, number]>;
  bar: BarConfig;
}

export interface OceanTableConfig {
  /** 1D phase table domain along x. Outside it the phase is extrapolated linearly. */
  phaseXMin: number;
  phaseXMax: number;
  phaseSize: number;
  /** z of the line whose depth profile is used for the phase integral (through the peak). */
  phaseLineZ: number;
  /** 2D depth field domain (must contain the whole bar). Outside it, depth = base profile. */
  fieldXMin: number;
  fieldXMax: number;
  fieldZMin: number;
  fieldZMax: number;
  fieldNx: number;
  fieldNz: number;
}

export interface OceanConfig {
  gravity: number;
  /** Depth-limited breaking index gamma: H_max = gamma * depth. */
  breakerIndex: number;
  /**
   * Maximum crest peakedness exponent p for shallow-water (cnoidal-like) waves. The swell
   * waveform is ((1 + cos psi) / 2)^p: p = 1 is a sinusoid, larger p gives narrow peaked
   * crests and long flat troughs. p ramps with the local Ursell number.
   */
  peakingMax: number;
  /** Ursell number H*L^2/h^3 where peaking starts / is fully developed. */
  ursellStart: number;
  ursellFull: number;
  /**
   * Maximum forward lean kappa (0..0.97) of waves about to break. The waveform phase psi is
   * related to the spatial phase phi by Kepler's equation  phi = psi + kappa*cos(psi - skewPhase),
   * which compresses the front face; kappa -> 1 makes the face vertical.
   */
  skewMax: number;
  /** Shifts the steepest point of the face toward the crest (radians, negative = higher). */
  skewPhase: number;
  /** Depth floor used for dispersion / shoaling maths. */
  minDepth: number;
  swells: SwellConfig[];
  wind: WindWaveConfig;
  bathymetry: BathymetryConfig;
  tables: OceanTableConfig;
}

/** A set of ~5 bigger waves followed by a lull. Values are relative heights. */
export const PRIMARY_SET_PATTERN = [
  0.42, 0.38, 0.5, 0.44, 0.4, 0.52, 0.62, 0.8, 0.95, 1.0, 0.9, 0.97, 0.7, 0.5, 0.42, 0.36,
];
export const SECONDARY_SET_PATTERN = [
  0.9, 0.7, 1.0, 0.8, 0.6, 0.95, 0.75, 0.85, 1.0, 0.65, 0.8, 0.9, 0.7, 1.0, 0.85, 0.6,
];

export const DEFAULT_BATHYMETRY: BathymetryConfig = {
  profile: [
    [-2000, 34],
    [-1000, 30],
    [-500, 22],
    [-250, 11],
    [-120, 6.0],
    [-40, 4.0],
    [40, 3.0],
    [80, 1.9],
    [110, 0.5],
    [125, 0.0],
    [160, -2.2],
    [260, -6.0],
    [600, -14.0],
  ],
  bar: {
    enabled: true,
    x0: -20,
    sweep: 0.7,
    roundness: 8,
    peakDepth: 1.4,
    depthSlope: 0.006,
    width: 26,
    fadeStartZ: 170,
    fadeEndZ: 250,
  },
};

export const DEFAULT_OCEAN_CONFIG: OceanConfig = {
  gravity: 9.81,
  breakerIndex: 0.78,
  peakingMax: 2.6,
  ursellStart: 8,
  ursellFull: 120,
  skewMax: 0.9,
  skewPhase: -0.35,
  minDepth: 0.15,
  swells: [
    { period: 11, height: 1.1, directionDeg: -6, phase: 0, setPattern: PRIMARY_SET_PATTERN },
    { period: 7.5, height: 0.25, directionDeg: 14, phase: 1.7, setPattern: SECONDARY_SET_PATTERN },
  ],
  wind: {
    count: 12,
    speed: 4.0,
    directionDeg: 25,
    spreadDeg: 45,
    minWavelength: 1.2,
    maxWavelength: 16,
    amplitudeScale: 0.7,
    choppiness: 0.8,
    steepnessBudget: 0.22,
    seed: 1337,
  },
  bathymetry: DEFAULT_BATHYMETRY,
  tables: {
    phaseXMin: -1600,
    phaseXMax: 400,
    phaseSize: 2048,
    phaseLineZ: 0,
    fieldXMin: -260,
    fieldXMax: 200,
    fieldZMin: -320,
    fieldZMax: 320,
    fieldNx: 512,
    fieldNz: 512,
  },
};

/** Deep clone so callers can mutate a config (e.g. from a debug GUI) without touching defaults. */
export function cloneOceanConfig(c: OceanConfig): OceanConfig {
  return JSON.parse(JSON.stringify(c)) as OceanConfig;
}
