import setuptools
from distutils.core import setup

if __name__ == '__main__':
     setup(
        name='ddo',
        version='0.1',
        author='Merlin Dugot',
        author_email='merlin.dugot@ghelia.com',
        packages=setuptools.find_packages(),
        description='Discovery of Deep Options',
        url='https://github.com/ghelia/prj-sie-autogameqa-ddo',
        # long_description=open('README.md').read(),
        install_requires=[
            "protobuf==3.20.1",
            "pygame",
            "torch",
            "gym",
            "numpy",
            "tqdm",
            "mypy",
            "tensorboard==2.7.0",
            "pytest",
            "pandas",
            "torchvision",
            "Pillow"
        ]
    )
