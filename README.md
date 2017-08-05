# pytorch-nips2017-attack-example

This is a baseline targeted (or untargeted) attack that works within the Cleverhans (https://github.com/tensorflow/cleverhans) framework for the NIPS-2017 adversarial competition. The attacks are modeled after the 'basic iterative' / 'itarative FGSM' attack mentioned in https://arxiv.org/abs/1611.01236 and https://arxiv.org/abs/1705.07204 (among others).

The default setup is to run a targeted L-inifity norm variant of the targeted attack with 10 steps. L1 or L2 based attacks seem to require around 40-50 steps with the current code to perform a reasonable attack.

To run:
1. Setup and verify cleverhans nips17 adversarial competition example environment
2. Clone this repo
3. Run ./download_checkpoint.sh to download the inceptionv3 checkpoint from torchvision model zoo
4. Symbolic link the folder this repo was clone into into the cleverhans 'examples/nips17_adversarial_competition/sample_targeted_attacks/' folder
5. Run run_attacks_and_defenses.sh and ensure '--gpu' flag is added


To switch to L2 or L1 norm or to perform a non-targeted attack, command line args in the run_attack.sh script need modification.

L1: 
```
python attack.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --steps 50 \
  --targeted \
  --norm 1 \
  --checkpoint_path=inception_v3_google-1a9a5a14.pth
```

L2:
```
python attack.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --steps 42 \
  --targeted \
  --norm 2 \
  --checkpoint_path=inception_v3_google-1a9a5a14.pth
```

