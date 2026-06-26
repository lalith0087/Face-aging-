# Fast-AgingGAN

A lightweight GAN for face aging. Given a face photo, the model generates an aged version of the same face.

## Setup

```bash
pip install -r requirements.txt
```

The pretrained generator weights are included at `pretrained_model/state_dict.pth`.

## Inference

Put face images (`.jpg` / `.png`) in a folder and run:

```bash
python infer.py --image_dir sample_images
```

This loads the generator, runs each image through it, and saves a before/after grid to `mygraph.png`.

## Training

Configure dataset paths and hyperparameters in `configs/aging_gan.yaml`, then run:

```bash
python main.py
```

Training uses PyTorch Lightning; see `gan_module.py` for the GAN training loop and `models.py` for the generator/discriminator architectures.

## Project layout

- `infer.py` — run inference on a folder of images
- `main.py` — train the GAN
- `gan_module.py` — Lightning module with the training/optimization logic
- `models.py` — generator and discriminator architectures
- `dataset.py` — dataset loading
- `preprocessing/` — scripts to preprocess the CACD and UTKFace datasets
- `pretrained_model/` — pretrained generator weights
