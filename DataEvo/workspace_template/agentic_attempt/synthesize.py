import argparse
import asyncio
import json
from pathlib import Path

from DataEvo.runtime.agentic_synthesis import synthesize

p = argparse.ArgumentParser()
p.add_argument('--context', required=True)
p.add_argument('--output', required=True)
args = p.parse_args()
asyncio.run(synthesize(json.loads(Path(args.context).read_text()), args.output))
