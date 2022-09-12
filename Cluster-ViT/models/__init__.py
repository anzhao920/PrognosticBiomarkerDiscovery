from .mortality_ViT import build_mortality_ViT


def build_model(args):
    return build_mortality_ViT(args)
