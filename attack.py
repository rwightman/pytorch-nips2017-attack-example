"""Attack loop
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import torch
import torchvision
import torch.utils.data as data

from scipy.misc import imsave
from dataset import Dataset, default_inception_transform


def run_attack(args, attack):
    assert args.input_dir

    if args.targeted:
        dataset = Dataset(
            args.input_dir,
            transform=default_inception_transform(args.img_size))
    else:
        dataset = Dataset(
            args.input_dir,
            target_file='',
            transform=default_inception_transform(args.img_size))

    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False)

    model = torchvision.models.inception_v3(pretrained=False, transform_input=False)
    if not args.no_gpu:
        model = model.cuda()

    if args.checkpoint_path is not None and os.path.isfile(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path)
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        print("Error: No checkpoint found at %s." % args.checkpoint_path)

    model.eval()

    for batch_idx, (input, target) in enumerate(loader):
        if not args.no_gpu:
            input = input.cuda()
            target = target.cuda()

        input_adv = attack.run(model, input, target, batch_idx)

        start_index = args.batch_size * batch_idx
        indices = list(range(start_index, start_index + input.size(0)))
        for filename, o in zip(dataset.filenames(indices, basename=True), input_adv):
            output_file = os.path.join(args.output_dir, filename)
            imsave(output_file, (o + 1.0) * 0.5, format='png')
