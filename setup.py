from setuptools import setup, find_packages
import os

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='recommender_system',
    version='1.0.0',
    author='Magdalena Borettini',
    author_email='magdalena.borettini@gmail.com',
    description='System rekomendacyjny oparty o model filtrowania kolaboratywnego',
    keywords='ml model recommeder_system collaborative_filtering',
    url='https://github.com/mborettini/recommender-system.git',
    install_requires=required,
    packages=find_packages(),
)
