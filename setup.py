from setuptools import find_packages, setup

setup(name="bayesee",
      version="0.0.1",
      description="Bayesian searcher in visual tasks",
      author="Anqi Zhang",
      author_email='anqikylin@gmail.com',
      platforms=["any"],
      license="GNU GPLv3",
      url="http://github.com/An-qi-Zhang/bayesee",
      packages=find_packages(),
      long_description=open('README.md').read() + "\n\n" + open('CHANGELOG.md').read(),
      classifiers=["Development Status :: Developing",
                   "Intended Audience :: Science/Research",
                   "Programming Language :: Python :: 3 "]
      )
