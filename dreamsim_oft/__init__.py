"""
[Begin Work Zone]
dreamsim_oft -- a standalone DreamSim-style perceptual metric trained with
OFTv2 adapters and the adv_optm SinkSGD_adv optimizer.

Why standalone: OneTrainer is diffusers-based and has no notion of triplet-ranking
perceptual-metric training, and DreamSim's own train.py needs pytorch_lightning +
wandb + torchmetrics, none of which are installed in this venv. So we reuse the
two things that matter -- OFTv2's rotation math and adv_optm -- and own the loop.
[End Work Zone]
"""

__version__ = "0.1.0"
