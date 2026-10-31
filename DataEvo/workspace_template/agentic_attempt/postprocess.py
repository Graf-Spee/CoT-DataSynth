import argparse

from DataEvo.runtime.agentic_synthesis import postprocess

p = argparse.ArgumentParser()
p.add_argument('--context', required=True)
p.add_argument('--input', required=True)
p.add_argument('--output', required=True)
args = p.parse_args()
postprocess(args.input, args.output)
