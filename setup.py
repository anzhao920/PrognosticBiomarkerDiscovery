from setuptools import find_packages, setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

def get_extensions():
 
    return extensions


if __name__ == "__main__":
    extensions = [
        CUDAExtension(
            "broadcast",
            sources=[
                "Cluster-ViT/models/extensions/broadcast.cu"
            ],
            extra_compile_args=["-arch=compute_50"]
        ),
        CUDAExtension(
            "weighted_sum",
            sources=[
                "Cluster-ViT/models/extensions/weighted_sum.cu"
            ],
            extra_compile_args=["-arch=compute_50"]
        )
    ]

    setup(
        name="clutering-Transformer",
        packages=find_packages(),
        ext_modules=extensions,
        cmdclass={"build_ext": BuildExtension},
        install_requires=["torch"]
    )
