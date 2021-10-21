from distutils.core import setup
setup(
  name = 'biotuner',         # How you named your package folder (MyLib)
  packages = ['biotuner'],   # Chose the same as "name"
  version = '0.0',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Time series harmonic analysis for adaptive tuning systems and microtonal exploration',   # Give a short description about your library
  author = 'Antoine Bellemare',                   # Type in your name
  author_email = 'antoine.bellemare9@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/antoinebellemare/biotuner',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/AntoineBellemare/biotuner/archive/refs/tags/v0.0.tar.gz',    # I explain this later on
  keywords = ['biosignal', 'harmony', 'tuning', 'eeg', 'microtonality', 'music', 'time series'],   # Keywords that define your package best
  install_requires=[
          'numpy',
          'matplotlib',
          'seaborn',
          'pygame'
          'pytuning'
          'mne'
          'bottleneck'
          'pyACA'
          'pactools'
          'colorednoise'
          'fooof'
          'emd'
          'emd-signal',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.8'
  ],
)