import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--steps',   help='choose the model steps', type=int)
parser.add_argument('--power',   help='choose the power: -3.0, 0.0, 3.0, 6.0', type=float)
args = parser.parse_args()

print(args.steps)
print(args.power)