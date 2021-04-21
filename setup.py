from setuptools import setup, find_packages
import codecs
import os.path


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="tscv",
    version=get_version("tscv/__init__.py"),
    author="Wenjie Zheng",
    author_email="work@zhengwenjie.net",
    description="Time series cross-validation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/WenjieZ/TSCV",
    license='new BSD',
    keywords='model selection, hyperparameter optimization, backtesting',
    packages=find_packages(),
    classifiers=['Intended Audience :: Science/Research',
                 'Intended Audience :: Developers',
                 'License :: OSI Approved :: BSD License',
                 'Programming Language :: Python',
                 'Topic :: Software Development',
                 'Topic :: Scientific/Engineering',
                 'Development Status :: 5 - Production/Stable',
                 'Operating System :: Microsoft :: Windows',
                 'Operating System :: POSIX',
                 'Operating System :: Unix',
                 'Operating System :: MacOS',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'Programming Language :: Python :: 3.9',
                 ],
    python_requires=">=3.6",
    install_requires=['numpy>=1.13.3', 'scikit-learn>=0.22']
)
