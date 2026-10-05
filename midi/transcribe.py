"""Audio -> MIDI with Spotify's basic-pitch (Apache-2.0).

Runs in its own small Python (midi/.venv, Python 3.10, created by tools/install-midi.bat):
basic-pitch needs older libraries than the app. samplegen calls it as a subprocess:

    python transcribe.py --audio loop.wav --out loop.mid [--bpm 140]

Prints one JSON line on success: {"notes": <count>, "out": "<path>"}.
"""

import argparse
import json
import sys
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--bpm", type=float, default=120.0, help="tempo written into the MIDI file")
    p.add_argument("--onset", type=float, default=0.5, help="higher = fewer, surer notes")
    p.add_argument("--frame", type=float, default=0.3)
    p.add_argument("--min-note-ms", type=float, default=80.0)
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    from basic_pitch.inference import predict  # slow import: after argument checks

    _, midi, notes = predict(
        args.audio, onset_threshold=args.onset, frame_threshold=args.frame,
        minimum_note_length=args.min_note_ms, midi_tempo=args.bpm,
    )
    out = Path(args.out)
    tmp = out.with_name(out.name + ".part")
    midi.write(str(tmp))
    tmp.replace(out)
    print(json.dumps({"notes": len(notes), "out": str(out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
