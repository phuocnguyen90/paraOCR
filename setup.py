# setup.py
from setuptools import setup, find_packages
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="paraocr",
    version="2.0.1",
    package_dir={"": "src"},
    packages=find_packages(where="src", include=["paraocr", "paraocr.*"]),
    author="Phuoc Nguyen",
    description="A high-performance, parallel OCR library for processing local files.",
    long_description=long_description,
    long_description_content_type='text/markdown',

    package_data={
        "paraocr": ["vi_full.txt"], 
    },
    include_package_data=True,

    install_requires=[
        "easyocr",
        "torch",
        "torchvision",
        "torchaudio",
        "PyMuPDF",
        "tqdm",
        "Pillow",
        "numpy"
    ],
    entry_points={
        'console_scripts': [
            'paraocr=paraocr.cli:main',
        ],
    },
)