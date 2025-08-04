from setuptools import setup, find_packages

setup(
    name="angualiberto-bioinformatics-system",
    version="2.0.0",
    author="AnguaLiberto",
    author_email="angualiberto@biotech.lab",
    description="A comprehensive bioinformatics and protein analysis system.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/angualiberto-bioinformatics-system",
    packages=find_packages(where='src'),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "matplotlib",
        "seaborn",
        "reportlab",
        "flask",
        "plotly"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)