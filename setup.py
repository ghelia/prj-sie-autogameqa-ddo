import setuptools
from distutils.core import setup

if __name__ == '__main__':
     setup(
        name='ddo',
        version='0.1',
        author='Merlin Dugot',
        author_email='merlin.dugot@ghelia.com',
        packages=setuptools.find_packages(),
        description='Deep Discovery of Options',
        url='https://github.com/ghelia/prj-sie-autogameqa-ddo',
        long_description=open('README.txt').read(),
        install_requires=[
            "torch",
            "gym",
            "numpy",
            "tqdm",
            "mypy",
            "tensorboard",
            "pytest"
        ]
    )
