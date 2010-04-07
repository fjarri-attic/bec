try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

DESCRIPTION = """Helper classes for data processing"""

CLASSIFIERS = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: GNU General Public License (GPL)",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Software Development :: Libraries :: Python Modules",
]

setup(
    name="datahelpers",
    version="0.0.1",
    description=DESCRIPTION,
    author="Bogdan Opanchuk",
    author_email="mantihor@gmail.com",
    py_modules=["datahelpers"],
    classifiers=CLASSIFIERS,
)
