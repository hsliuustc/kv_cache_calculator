from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

setup(
    name="kv_cache_calculator",
    version="0.1.0",
    description="A Python package for calculating KV cache sizes for transformer-based language models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/kv_cache_calculator",
    author="Your Name",
    author_email="your.email@example.com",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="llm, transformer, kv cache, memory optimization",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8, <4",
    install_requires=[
        "PyYAML>=6.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=8.2.0",
            "pytest-cov>=5.0.0",
            "black>=24.0.0",
            "flake8>=7.0.0",
            "mypy>=1.0.0",
            "pre-commit>=3.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "kv-cache-calc=kv_cache_calculator.cli:main",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/yourusername/kv_cache_calculator/issues",
        "Source": "https://github.com/yourusername/kv_cache_calculator",
    },
)
