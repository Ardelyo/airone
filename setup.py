from setuptools import setup, find_packages
import os

# Read version
version_dict = {}
with open(os.path.join("airone", "__version__.py")) as f:
    exec(f.read(), version_dict)

setup(
    name="airone",
    version=version_dict["__version__"],
    description="Intelligent Semantic Compression Platform",
    long_description="AirOne is a multi-layered intelligent compression platform.",
    long_description_content_type="text/markdown",
    author="AirOne Team",
    author_email="hello@airone.io",
    url="https://github.com/airone/airone",
    packages=find_packages(exclude=["tests", "docs", "examples"]),
    install_requires=[
        "numpy>=1.21.0",
        "click>=8.0.0",
        "Pillow>=9.0.0",
        "pypdf>=3.0.0",
        "zstandard>=0.19.0",
        "brotli>=1.0.9",
        "msgpack>=1.0.0",
        "xxhash>=3.0.0",
    ],
    extras_require={
        "ml": [
            "torch>=2.0.0",
            "torchvision>=0.15.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=6.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "airone=airone.cli.main:cli",
        ],
    },
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
