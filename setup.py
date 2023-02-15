from setuptools import setup

setup(
      name='pyXEOL',
      url='https://github.com/narahma2/pyXEOL',
      author='Naveed Rahman',
      author_email='naveed@anl.gov',
      packages=['pyXEOL', 'pyXEOL.xeol'],
      package_dir={'':'src'},
      version='0.0.1',
      description='XEOL analysis package',
      long_description=open('README.md').read()
      )
