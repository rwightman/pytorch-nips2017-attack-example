"""Sample Pytorch attack.

This defense loads inception v3 checkpoint and classifies all images
using loaded checkpoint.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import argparse
import numpy as np

import torch
import torch.autograd as autograd
import torch.utils.data as data
import torchvision
import torchvision.datasets.folder
import torchvision.transforms as transforms

from torch.autograd.gradcheck import zero_gradients
from scipy.misc import imsave
from dataset import Dataset

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
parser.add_argument('--step_eps', type=float, default=0.0,
                    help='Per step epsilon, defaults to epsilon/steps')
parser.add_argument('--norm', default='inf', type=float,
                    help='Gradient norm.')
parser.add_argument('--targeted', action='store_true', default=False,
                    help='Targeted attack')
parser.add_argument('--no_gpu', action='store_true', default=False,
                    help='disables GPU training')


class LeNormalize(object):
    """Normalize to -1..1 in Google Inception style
    """
    def __call__(self, tensor):
        for t in tensor:
            t.sub_(0.5).mul_(2.0)
        return tensor


def l2norm(x):
    norm = x * x
    # silly PyTorch, when will you get reducing sums/means?
    for a in [3, 2, 1]:
        norm = norm.sum(a, keepdim=True)
    return norm.sqrt_()


def l1norm(x):
    norm = x.abs()
    for a in [3, 2, 1]:
        norm = norm.sum(a, keepdim=True)
    return norm


def main():
    args = parser.parse_args()
    assert args.input_dir

    debug = True
    eps = 2.0 * args.max_epsilon / 255.0
    num_steps = args.steps
    if not args.step_eps:
        if args.norm == float('inf'):
            step_eps = eps / num_steps
        else:
            # Don't bother epsilon step (down) scaling for non infinity norms, otherwise
            # we never get anywhere with the inifinty-norm clipping constraint applied.
            # If anything, we may want to scale up to get a reasonably potent attack
            # in less steps.
            if args.norm == 1:
                step_eps = 500.0  # L1 needs a lot of (arbitrary) love
            else:
                step_eps = 1.0
    else:
        step_eps = args.step_eps

    if args.targeted:
        dataset = Dataset(args.input_dir)
    else:
        dataset = Dataset(args.input_dir, target_file='')

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False)

    tf = transforms.Compose([
        transforms.Scale(args.img_size),
        transforms.CenterCrop(args.img_size),
        transforms.ToTensor(),
        LeNormalize(),
    ])
    dataset.set_transform(tf)

    model = torchvision.models.inception_v3(pretrained=False, transform_input=False)
    loss_fn = torch.nn.CrossEntropyLoss()
    if not args.no_gpu:
        model = model.cuda()
        loss_fn = loss_fn.cuda()
    model.eval()

    if args.checkpoint_path is not None and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("Error: No checkpoint found at %s." % args.checkpoint_path)

    for batch_idx, (input, target) in enumerate(loader):
        if not args.no_gpu:
            input = input.cuda()
            target = target.cuda()
        input_var = autograd.Variable(input, volatile=False, requires_grad=True)
        target_var = autograd.Variable(target, volatile=False)

        step = 0
        while step < num_steps:
            zero_gradients(input_var)
            output = model(input_var)
            if not args.targeted and not step:
                # for non-targeted, we'll move away from most likely
                target_var.data = output.data.max(1)[1]
            loss = loss_fn(output, target_var)
            loss.backward()

            # normalize and scale gradient
            if args.norm == 2:
                normed_grad = step_eps * input_var.grad.data / l2norm(input_var.grad.data)
            elif args.norm == 1:
                normed_grad = step_eps * input_var.grad.data / l1norm(input_var.grad.data)
            else:
                # infinity-norm
                normed_grad = step_eps * torch.sign(input_var.grad.data)

            # perturb current input image by normalized and scaled gradient
            if args.targeted:
                step_adv = input_var.data - normed_grad
            else:
                step_adv = input_var.data + normed_grad

            # calculate total adversarial perturbation from original image and clip to epsilon constraints
            total_adv = step_adv - input
            if args.norm == 2:
                # total_adv = eps * total_adv / l2norm(total_adv)
                total_adv = torch.clamp(total_adv, -eps, eps)
            elif args.norm == 1:
                # total_adv = eps * total_adv / l1norm(total_adv)
                total_adv = torch.clamp(total_adv, -eps, eps)
            else:
                # infinity-norm
                total_adv = torch.clamp(total_adv, -eps, eps)

            if debug:
                print('batch', batch_idx, 'step', step, total_adv.mean(), total_adv.min(), total_adv.max())
                sys.stdout.flush()

            # apply total adversarial perturbation to original image and clip to valid pixel range
            input_adv = input + total_adv
            input_adv = torch.clamp(input_adv, -1.0, 1.0)
            input_var.data = input_adv
            step += 1

        input_adv = input_adv.permute(0, 2, 3, 1)
        start_index = args.batch_size * batch_idx
        indices = list(range(start_index, start_index + input_var.size(0)))
        for filename, o in zip(dataset.filenames(indices, basename=True), input_adv.cpu().numpy()):
            output_file = os.path.join(args.output_dir, filename)
            imsave(output_file, (o + 1.0) * 0.5, format='png')


if __name__ == '__main__':
    main()
