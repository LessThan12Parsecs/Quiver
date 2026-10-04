/** Render module public API (see each file's header for details). */
export { OceanScene, type OceanSceneOptions } from './OceanScene';
export { Environment, sunDirectionFromAngles, type CloudOptions, type EnvironmentOptions } from './Environment';
export {
  OCEAN_QUALITY_RATIO,
  OCEAN_SHADING,
  OceanMesh,
  WaterLook,
  type OceanMeshOptions,
  type OceanMeshStats,
  type OceanQuality,
  type OceanShading,
  type OceanShadingPreset,
} from './OceanMesh';
export { MAX_TILE_COMPONENTS, RippleTile, type RippleTileOptions } from './DetailNormals';
export { BeachMesh, type BeachMeshOptions } from './BeachMesh';
export { DEFAULT_SPRAY_REGIONS, Spray, type SprayOptions, type SprayRegion } from './Spray';
export { CDLOD_MORPH_START, CdlodSelector, PatchKind, type CdlodOptions } from './cdlod';
export { DEFAULT_SKY_PARAMS, SkyModel, luminance, type SkyParams } from './skyModel';
export { ENV_GLSL, NOISE_GLSL } from './shaders/common';
export {
  BOARD_LOOKS,
  BoardMesh,
  buildFinGeometry,
  buildHullGeometry,
  type BoardLook,
  type BoardMeshOptions,
} from './BoardMesh';
export { RIDER_RENDER_ORDER, RiderMesh, solveTwoBone, type RiderMeshOptions } from './RiderMesh';
