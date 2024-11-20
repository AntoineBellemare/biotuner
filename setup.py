from setuptools import setup
import os

# Dynamically locate requirements.txt
current_dir = os.path.abspath(os.path.dirname(__file__))
requirements_path = os.path.join(current_dir, 'requirements.txt')

if os.path.exists(requirements_path):
    with open(requirements_path) as f:
        install_requires = f.read().splitlines()
else:
    install_requires = []

setup(
  name='biotuner',
  packages=['biotuner'],
  version='0.0.12',
  license='MIT',
  description='Time series harmonic analysis for adaptive tuning systems and microtonal exploration',
  author='Antoine Bellemare',
  author_email='antoine.bellemare9@gmail.com',
  url='https://github.com/antoinebellemare/biotuner',
  keywords=['biosignal', 'harmony', 'tuning', 'eeg', 'microtonality', 'music', 'time series'],
  install_requires=install_requires,  # Use the dynamically loaded install_requires
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)
