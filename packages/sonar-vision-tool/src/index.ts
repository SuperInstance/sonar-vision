import { z } from 'zod';

const SedimentEnum = z.enum(['mud', 'sand', 'gravel', 'rock', 'seagrass']);
const SeasonEnum = z.enum(['summer', 'winter']);
const ActionEnum = z.enum(['infer', 'physics', 'health']);

export const SonarVisionSchema = z.object({
  action: ActionEnum,
  depth: z.number().min(0).max(1000).optional(),
  depths: z.array(z.number()).optional(),
  chlorophyll: z.number().min(0).max(30).default(5.0),
  season: SeasonEnum.default('summer'),
  sediment: SedimentEnum.default('sand'),
  wavelength: z.number().min(300).max(700).default(480),
  salinity: z.number().min(0).max(45).default(35),
  frequency_khz: z.number().min(10).max(1000).optional(),
  timeout: z.number().min(100).max(30000).default(5000),
});

export type SonarVisionParams = z.infer<typeof SonarVisionSchema>;

export interface SonarFrame {
  depth: number;
  water_type: number;
  water_type_name: string;
  temperature: number;
  absorption: number;
  scattering: number;
  attenuation: number;
  visibility: number;
  seabed_reflectivity: number;
  sound_speed: number;
  refraction_deg: number;
}

export interface HealthResponse {
  status: string;
  uptime_seconds: number;
  version: string;
}

const SEDIMENT_REFLECT: Record<string, number> = {
  mud: 0.3, sand: 0.5, gravel: 0.7, rock: 0.85, seagrass: 0.2,
};

const WATER_TYPES: Record<number, string> = {
  0: 'Coastal', 1: 'Oceanic Type II', 2: 'Oceanic Type IB', 3: 'Clear Oceanic',
};

function computePhysics(depth: number, chl: number, season: string, sediment: string, wl: number, sal: number): SonarFrame {
  const wt = chl > 10 ? 0 : chl > 1 ? 1 : chl > 0.1 ? 2 : 3;
  const wa = wl / 1000;

  const absorption = wt <= 1
    ? 0.04 + 0.96 * Math.exp(-((wa - 0.42) ** 2) / (2 * 0.02 ** 2))
    : wt === 2
    ? 0.3 + 0.9 * Math.exp(-((wa - 0.48) ** 2) / (2 * 0.03 ** 2))
    : 0.02 + 0.51 * Math.exp(-((wa - 0.42) ** 2) / (2 * 0.015 ** 2));

  const ns = 0.002 * Math.pow(480 / wl, 4.3);
  const scattering = ns * Math.max(0.01, 1 - depth * 0.003);

  const tc = season === 'summer' ? 15 : 40;
  const tw = season === 'summer' ? 5 : 15;
  const st = season === 'summer' ? 22 : 8;
  const dt = 4;
  const temp = dt + (st - dt) * Math.exp(-((depth - tc) ** 2) / (2 * tw ** 2));

  const seabed = (SEDIMENT_REFLECT[sediment] || 0.5) * Math.exp(-depth * 0.003);
  const atten = absorption + scattering;
  const vis = Math.min(depth, 1.7 / Math.max(atten, 0.001));

  const ss = 1449.2 + 4.6 * temp - 0.055 * temp ** 2 + 0.00029 * temp ** 3
    + (1.34 - 0.01 * temp) * (sal - 35) + 0.016 * depth;

  const vRatio = ss / 1480;
  const theta = Math.sin(Math.PI / 6);
  const refraction = Math.asin(Math.min(1, theta / vRatio)) * (180 / Math.PI);

  return {
    depth, water_type: wt, water_type_name: WATER_TYPES[wt],
    temperature: Math.round(temp * 100) / 100,
    absorption: Math.round(absorption * 10000) / 10000,
    scattering: Math.round(scattering * 10000) / 10000,
    attenuation: Math.round(atten * 1000) / 1000,
    visibility: Math.round(vis * 100) / 100,
    seabed_reflectivity: Math.round(seabed * 10000) / 10000,
    sound_speed: Math.round(ss * 10) / 10,
    refraction_deg: Math.round(refraction * 100) / 100,
  };
}

export class SonarVisionTool {
  private endpoint: string;
  private timeout: number;

  constructor(opts: { endpoint?: string; timeout?: number } = {}) {
    this.endpoint = opts.endpoint || 'http://localhost:8080';
    this.timeout = opts.timeout || 5000;
  }

  async execute(params: SonarVisionParams): Promise<SonarFrame | SonarFrame[] | HealthResponse | null> {
    const validated = SonarVisionSchema.parse(params);

    switch (validated.action) {
      case 'physics': {
        const depth = validated.depth || 15;
        return computePhysics(depth, validated.chlorophyll, validated.season,
          validated.sediment, validated.wavelength, validated.salinity);
      }
      case 'infer': {
        const depths = validated.depths || [0, 15, 30, 50, 100];
        return depths.map(d => computePhysics(d, validated.chlorophyll, validated.season,
          validated.sediment, validated.wavelength, validated.salinity));
      }
      case 'health':
        return { status: 'ok', uptime_seconds: 0, version: '1.0.0' };
      default:
        return null;
    }
  }
}

export default SonarVisionTool;
