from setuptools import setup
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.txt").read_text()

setup(# the name must match the folder name 
name='bassi',
version='0.8',
long_description=long_description,
author='Francisco Romero',
author_email='francisco.romero@cimat.mx',
license='MIT',
packages=['bassi'],
install_requires=["alive-progress"],
zip_safe=False)

