import setuptools

with open("README.md", "r", encoding='utf8') as fh:
    long_description = fh.read()

requires = []
with open('requirements.txt', encoding='utf8') as f:
    for x in f.readlines():
        requires.append(f'{x.strip()}')


setuptools.setup(
    name="torch-analyzer",
    py_modules=["torch-analyzer"],
    version="1.4.2",
    author="Ziyi Dong",
    author_email="dzy7eu7d7@gmail.com",
    description="A torch model analyzer",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IrisRainbowNeko/torch-analyzer",
    packages=setuptools.find_packages(),
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.7',

    install_requires=requires
)
