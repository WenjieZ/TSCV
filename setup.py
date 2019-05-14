import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tscv",
    version="0.0.1",
    author="Wenjie Zheng",
    author_email="work@zhengwenjie.net",
    description="Time series cross-validation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/WenjieZ/TSCV",
    license='new BSD',
    packages=setuptools.find_packages(),
    classifiers=['Intended Audience :: Science/Research',
                 'Intended Audience :: Developers',
                 'License :: OSI Approved',
                 'Programming Language :: Python',
                 'Topic :: Software Development',
                 'Topic :: Scientific/Engineering',
                 'Operating System :: Microsoft :: Windows',
                 'Operating System :: POSIX',
                 'Operating System :: Unix',
                 'Operating System :: MacOS',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                  ],
    python_requires=">=3.5"
)
