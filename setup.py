import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name = 'BarTracking',
        version = 0.3,
        author = 'cluffa',
        author_email = 'alexcluff16@gmail.com',
        url = 'https://github.com/cluffa/bar_tracking',
        packages = ['BarTracking'],
        package_dir={'BarTracking': 'BarTracking'},
        package_data={'BarTracking': ['*.onnx']},
        test_suite = 'tests',
        install_requires = [
            'numpy',
            'scipy',
            'pandas',
            'onnxruntime-gpu',
            'torchvision',
            'opencv-python',
            'matplotlib',
            'segmentation-models-pytorch'
        ],
    )
    
    