from argparse import ArgumentParser
import yaml
from pytorch_lightning import Trainer
from gan_module import AgingGAN
import torch

def main():
    parser = ArgumentParser()
    parser.add_argument('--config', default='configs/aging_gan.yaml', help='Config to use for training')
    parser.add_argument('--metadata', required=True, help='Path to the metadata file')
    args = parser.parse_args()
    with open(args.metadata) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    print(torch.backends.mps.is_available())
    print(config)
    model = AgingGAN(config)
    trainer = Trainer(
        max_epochs=config['epochs'],
        devices=1,  # Use 1 GPU
        accelerator='mps'  # Specify the MPS accelerator for macOS
    )
    trainer.fit(model)

if __name__ == '__main__':
    main()
