"""
NonNULL Framework V1.0 - Package Installation Script
File: setup.py
"""

import os
from setuptools import setup, find_packages

readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = "The NonNULL Framework - Adaptive Meta-Control & Narrative Alignment"


def get_version() -> str:
    init_path = os.path.join("doctrine", "__init__.py")
    if os.path.exists(init_path):
        with open(init_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("__version__"):
                    return line.split("=")[1].strip().strip('"').strip("'")
    return "1.0.0"


install_requires = [
    "torch>=2.1.0",
    "transformers>=4.38.0",
    "peft>=0.9.0",
    "bitsandbytes>=0.42.0",
    "accelerate>=0.27.0",
    "datasets>=2.16.0",
    "pyyaml>=6.0",
    "tqdm>=4.65.0",
    "colorama>=0.4.6",
    "numpy>=1.24.0",
]

extras_require = {
    "memory": [
        "sentence-transformers>=2.3.0",
        "faiss-cpu>=1.7.4",
    ],
    "dev": [
        "pytest>=7.4.0",
    ],
    "monitoring": [
        "wandb>=0.16.0",
        "tensorboard>=2.15.0",
    ],
}
extras_require["all"] = sum(extras_require.values(), [])

setup(
    name="doctrine",
    version=get_version(),
    description="NonNULL Framework - Adaptive Meta-Control & Narrative-Based AI Alignment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Giang Nhan Nghia",
    packages=find_packages(exclude=["tests", "tests.*"]),
    py_modules=["run_training", "evaluate_genesis", "interactive_simulation"],
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "nonnull-train=run_training:main",
            "nonnull-eval=evaluate_genesis:main",
            "nonnull-sim=interactive_simulation:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)