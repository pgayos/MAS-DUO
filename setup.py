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
        "render": ["pygame>=2.5.0"],
        "train":  ["stable-baselines3>=2.2.0", "supersuit>=3.9.0"],
    },
)
