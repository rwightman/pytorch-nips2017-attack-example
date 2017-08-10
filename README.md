# pytorch-nips2017-attack-example

This is a baseline targeted (or untargeted) attack that works within the Cleverhans (https://github.com/tensorflow/cleverhans) framework for the NIPS-2017 adversarial competition. 

There are two types of attacks included, an iterative fast-gradient method, and a Carlini and Wagner L2 attack.

## Iterative Fast-Gradient

These attacks are modeled after the 'basic iterative' / 'itarative FGSM' attack mentioned in https://arxiv.org/abs/1611.01236 and https://arxiv.org/abs/1705.07204 (among others).

The default setup is to run a targeted L-inifity norm variant of the targeted attack with 10 steps. L1 or L2 based attacks seem to require around 40-50 steps with the current code to perform a reasonable attack.

## Carlini and Wagner L2

An implementation of the L2 variant of the attack described in this paper https://arxiv.org/abs/1608.04644 by Carlini and Wagner. Based on a reference implementation by Carlini at https://github.com/carlini/nn_robust_attacks and  https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks_tf.py

NOTE: I'm still verifying and experimenting with this attack. It takes MUCH longer (half a day) to run and produces much more subtle results that I'm having difficulty successfully transfering as a targeted attack to other models... 

## Usage

To run:
1. Setup and verify cleverhans nips17 adversarial competition example environment
2. Clone this repo
3. Run ./download_checkpoint.sh to download the inceptionv3 checkpoint from torchvision model zoo
4. Symbolic link the folder this repo was clone into into the cleverhans 'examples/nips17_adversarial_competition/sample_targeted_attacks/' folder
5. Run run_attacks_and_defenses.sh and ensure '--gpu' flag is added


To switch between attacks and alter parameters of the attack, command line args in the run_attack.sh script need modification.

Iterative non-targeted L1: 
```
python run_attack_iter.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --steps 50 \
  --norm 1 \
  --checkpoint_path=inception_v3_google-1a9a5a14.pth
```

Iterative targeted L2:
```
python run_attack_iter.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --steps 42 \
  --targeted \
  --norm 2 \
  --checkpoint_path=inception_v3_google-1a9a5a14.pth
```

Carlini and Wagner L2:
```
python run_attack_cwl2.py \
  --input_dir="${INPUT_DIR}" \
  --output_dir="${OUTPUT_DIR}" \
  --max_epsilon="${MAX_EPSILON}" \
  --targeted \
  --checkpoint_path=inception_v3_google-1a9a5a14.pth
```



