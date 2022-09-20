import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--power',   help='choose the power', type=float, nargs='+')
args = parser.parse_args()

print(args.power)