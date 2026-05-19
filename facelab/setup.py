from setuptools import setup, find_packages

setup(
    name="facialgui",
    version="1.0",
    python_requires=">=3.7",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "opencv-python",
        "PyQt5",
        "scikit-image",
        "scipy",
        "torch",
        "pyqtgraph",
        "h5py",
        "tqdm",
        "scikit-learn",
        "matplotlib"
    ],
)