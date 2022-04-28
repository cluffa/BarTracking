import setuptools

if __name__ == "__main__":
    setuptools.setup(
        setup_resuires = ['torch'],
        install_requires = [
            "setuptools",
            "numpy",
            "scipy",
            "pandas",
            "scikit-image",
            "torch",
            "torchvision",
            "opencv-python"
            ],
    )