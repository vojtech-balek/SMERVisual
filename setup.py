from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'SMER Visual classication with the use of LLMs.'
LONG_DESCRIPTION = ('Package implementing SMER method for classification and explanation of features in image '
                    'classification.')

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="smer_visual_classification",
    version=VERSION,
    author="Jason Dsouza",
    author_email="vojta.balek@gmail.com>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[],
    keywords=['python', 'smer', 'llms', 'explanable'],
    classifiers= [
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 3",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)