"""
Setup configuration for LightGBM package.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="lightgbm-scratch",
    version="0.1.0",
    author="KDD Project Team",
    description="Implementation of LightGBM from scratch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False,
    url="https://github.com/amineEssahraoui/KDD_projet",
    project_urls={
        "Documentation": "https://github.com/amineEssahraoui/KDD_projet/tree/main/docs",
        "Bug Reports": "https://github.com/amineEssahraoui/KDD_projet/issues",
        "Source Code": "https://github.com/amineEssahraoui/KDD_projet",
    },
)
