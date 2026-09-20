from setuptools import setup, find_packages

setup(
    name="mas-duo-logistics",
    version="0.1.0",
    description="Multi-agent logistics environment for RL (PettingZoo + Gymnasium)",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "gymnasium>=0.29.0",
        "pettingzoo>=1.24.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        # pygame-ce keeps the public ``pygame`` API and ships wheels for
        # current Python versions, including Python 3.14 on macOS.
        "render": ["pygame-ce>=2.5.0"],
        "train":  ["stable-baselines3>=2.2.0", "supersuit>=3.9.0"],
        "test":   ["pytest>=8.0", "pytest-cov>=5.0"],
    },
)
