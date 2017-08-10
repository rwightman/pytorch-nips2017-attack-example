"""Pytorch Iterate Fast-Gradient attack runner.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
from attacks import AttackIterative
from attack import run_attack

parser = argparse.ArgumentParser(description='Defence')
parser.add_argument('--input_dir', metavar='DIR',
                    help='Input directory with images.')
parser.add_argument('--output_dir', metavar='FILE',
                    help='Output directory to save images.')
parser.add_argument('--checkpoint_path', default=None,
                    help='Path to network checkpoint.')
parser.add_argument('--img_size', type=int, default=299, metavar='N',
                    help='Image patch size (default: 299)')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='Batch size (default: 32)')
parser.add_argument('--max_epsilon', type=int, default=16, metavar='N',
                    help='Maximum size of adversarial perturbation. (default: 16.0)')
parser.add_argument('--steps', type=int, default=10, metavar='N',
                    help='Number of steps to run attack for')
parser.add_argument('--step_alpha', type=float, default=0.0,
                    help='Per step scaling constant, defaults to epsilon/steps')
parser.add_argument('--norm', default='inf', type=float,
                    help='Gradient norm.')
parser.add_argument('--targeted', action='store_true', default=False,
                    help='Targeted attack')
parser.add_argument('--no_gpu', action='store_true', default=False,
                    help='Disable GPU training')
parser.add_argument('--debug', action='store_true', default=False,
                    help='Enable verbose debug output')


def main():
    args = parser.parse_args()
    attack = AttackIterative(
        targeted=args.targeted,
        max_epsilon=args.max_epsilon,
        norm=args.norm,
        step_alpha=args.step_alpha,
        num_steps=args.steps,
        cuda=not args.no_gpu,
        debug=args.debug)

    run_attack(args, attack)

if __name__ == '__main__':
    main()
