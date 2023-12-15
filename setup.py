import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="CVor",
    version="0.0.7",
    author="chenxingyan94",
    author_email="xychen@swufe.edu.cn",
    description="A simple implementation of control variates operator (CVor)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/uglyghost/CVOR.git",
    packages=setuptools.find_packages(),
    install_requires=[
        'torch',
        'torchvision',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)