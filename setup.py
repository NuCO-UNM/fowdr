import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fowdr",
    version="1.1",
    author="Huaiyu Duan",
    author_email="duan@unm.edu",
    description="A package that computes the dispersion relations of the fast flavor oscillations wave in dense neutrino gases.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mikeduan/fowdr",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[ 'numpy', 'scipy' ]
)