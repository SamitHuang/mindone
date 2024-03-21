"""
Train AutoEncoder KL-reg with GAN loss
"""


def main(args):
    # create models: autoencoder, discrimintor
    generator = build_model_from_config(vae_config)


if __name__ == '__main__':
    main()
