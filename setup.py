from setuptools import setup, find_packages

setup(
    name="neuroarch",
    version="1.0.0",
    description="NeuroArch: SNN + MARL for Building Energy Optimization",
    author="Mohamed Ali, Saher Elsayed, Ts. Dr. Khairi Azhar Aziz",
    author_email="selsayed@seas.upenn.edu",
    url="https://github.com/NeuroArch-Lab/NeuroArch",
    packages=find_packages(exclude=["tests*", "notebooks*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.2",
        "numpy>=1.26",
        "pandas>=2.1",
        "gymnasium>=0.29",
        "pyyaml>=6.0",
    ],
    extras_require={
        "server": ["fastapi", "uvicorn[standard]", "websockets"],
        "dev":    ["pytest", "pytest-cov", "ruff", "black", "isort"],
        "bim":    ["ifcopenshell"],
    },
    classifiers=[
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Human Machine Interfaces",
    ],
)
