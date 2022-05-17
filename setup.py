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
        url='https://wiki.ghelia.dev/626667f0ece6fa80e19898dd#%E6%8F%90%E6%A1%88%EF%BC%91%EF%BC%9A%E3%83%95%E3%82%A9%E3%83%AF%E3%83%BC%E3%83%89-%E3%83%90%E3%83%83%E3%82%AF%E3%83%AF%E3%83%BC%E3%83%89%E3%82%A2%E3%83%AB%E3%82%B4%E3%83%AA%E3%82%BA%E3%83%A0%E3%81%AB%E5%9F%BA%E3%81%A5%E3%81%8F%E3%82%AA%E3%83%97%E3%82%B7%E3%83%A7%E3%83%B3%E6%8E%A2%E7%B4%A2',
        long_description=open('README.txt').read(),
        install_requires=[
            "torch",
            "gym",
            "numpy",
            "tqdm",
            "mypy",
            "tensorboard"
        ]
    )
