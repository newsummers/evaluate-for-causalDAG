from setuptools import setup, find_packages

setup(
    name="evaluate-causal-dag",
    version="0.1.0",
    description="Multi-dimensional causal DAG accuracy evaluation",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "networkx>=3.0",
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.10",
    ],
    extras_require={
        "dev": ["pytest>=7.0"],
    },
)
