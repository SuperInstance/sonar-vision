"""SonarVision CLI — ping, dive, serve."""

import argparse
import json
import sys

from .physics import compute_physics, dive_profile


def main():
    parser = argparse.ArgumentParser(description='SonarVision Physics CLI')
    parser.add_argument('--depth', type=float, default=15.0, help='Depth in meters')
    parser.add_argument('--chlorophyll', type=float, default=5.0)
    parser.add_argument('--season', choices=['summer', 'winter'], default='summer')
    parser.add_argument('--sediment', choices=['mud', 'sand', 'gravel', 'rock', 'seagrass'],
                        default='sand')
    parser.add_argument('--dive', action='store_true', help='Compute dive profile')
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--end', type=float, default=100)
    parser.add_argument('--step', type=float, default=5)
    parser.add_argument('--format', choices=['text', 'json'], default='text')
    parser.add_argument('--serve', action='store_true', help='Start WebSocket server')
    parser.add_argument('--port', type=int, default=8081, help='Server port')
    parser.add_argument('--rate', type=float, default=5.0, help='Stream rate (Hz)')
    
    args = parser.parse_args()
    
    if args.serve:
        return _serve(args.port, args.rate)
    
    if args.dive:
        results = dive_profile(int(args.start), int(args.end), int(args.step),
                               args.chlorophyll, 0 if args.season == 'summer' else 1)
        if args.format == 'json':
            print(json.dumps({'profile': results, 'count': len(results)}, indent=2))
        else:
            print(f"{'Depth':>5} {'Type':>18} {'Temp':>6} {'Vis':>6} {'Sound':>7}")
            print('-' * 55)
            for r in results:
                print(f"{r['depth']:5.0f}m {r['water_type_name']:>18} "
                      f"{r['temperature']:6.1f}C {r['visibility']:6.1f}m "
                      f"{r['sound_speed']:7.1f}m/s")
    
    else:
        result = compute_physics(args.depth, args.chlorophyll,
                                 0 if args.season == 'summer' else 1)
        if args.format == 'json':
            print(json.dumps(result, indent=2))
        else:
            print(f"=== SONAR PING @ {result['depth']:.0f}m ===")
            print(f"  Water Type : {result['water_type_name']}")
            print(f"  Temperature: {result['temperature']:.1f}C")
            print(f"  Absorption : {result['absorption']:.4f} m^-1")
            print(f"  Scattering : {result['scattering']:.4f} m^-1")
            print(f"  Attenuation: {result['attenuation']:.3f} m^-1")
            print(f"  Visibility : {result['visibility']:.1f}m")
            print(f"  Seabed Refl: {result['seabed_reflectivity']:.3f}")
            print(f"  Sound Speed: {result['sound_speed']:.0f} m/s")
            print(f"  Refraction : {result['refraction_deg']:.1f} deg")


def _serve(port, rate):
    """Start WebSocket streaming server."""
    try:
        from .streaming import StreamingServer
        import asyncio
        server = StreamingServer(port=port, rate_hz=rate)
        asyncio.run(server.start())
    except ImportError:
        print("Streaming server not available. Install with: pip install sonar-vision-physics[streaming]")
        sys.exit(1)


if __name__ == '__main__':
    main()
