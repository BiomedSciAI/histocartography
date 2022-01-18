"""Install package."""
import re
import os
import sys
import subprocess
import traceback
from setuptools import setup, find_packages, Command
from setuptools.command.bdist_egg import bdist_egg as _bdist_egg
from setuptools.command.develop import develop as _develop
from distutils.command.build import build as _build

VCS_REQUIREMENTS = []
PYPI_REQUIREMENTS = [
    "torch>=1.3.1",
    "tqdm>=4.35.0",
    "pandas>=0.24.2",
    "matplotlib>=3.1.1",
    "h5py>=2.9.0",
    "scikit-learn>=0.22",
    "seaborn>=0.9.0",
    "torchvision>=0.4.0",
    "pillow>=7.2.0",
    "opencv-python>=3.4.8.29",
    "scikit-image>=0.17.2",
    "dgl==0.4.3.post2",
    "PyYAML>=5.1.2",
]
regex = re.compile(r"(git|svn|hg|bzr)\+")
if os.path.exists("requirements.txt"):
    for line in open("requirements.txt"):
        if regex.match(line):
            VCS_REQUIREMENTS.append(line.strip())
        else:
            PYPI_REQUIREMENTS.append(line.strip())


def install(package):
    """
    Install a specific package using pip.

    Installation with `pip install` will ignore location options like
    --user or --prefix= for VCS dependencies. In this case it is attempted
    to install to site-packages location and on failure to user-site.

    Installation location can only be defined when calling setup.py directly!
    e.g:
    [PYTHONUSERBASE=/path/to/install] python3 setup.py install --user package
    python3 setup.py install --prefix='/path/to/install' package
    """
    # `pip install` will result in `setup.py bdist_wheel`
    # location options are lost in pip
    install_command = [sys.executable, "-m", "pip", "install"]
    # try to get users install_options in case of `setup.py install` sys.argv
    if "--user" in sys.argv:
        install_command.append("--user")
    else:
        # can't combine user with prefix, exec_prefix/home, install_(plat)base
        try:
            install_command.append(next(filter(lambda x: "--prefix=" in x, sys.argv)))
        except StopIteration:
            pass
    try:
        subprocess.check_call(install_command + [package])
    except subprocess.CalledProcessError as exc:
        print("setup.py sys.argv are:\n", sys.argv)
        print(
            "Installation attempt failed with command {}\n"
            "Trying to install with --user now.".format(install_command)
        )
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--user", package]
        )
    except Exception as exc:
        print("setup.py sys.argv are:\n", sys.argv)
        traceback.print_exc()
        raise exc


class install_dependencies_from_vcs(Command):
    """
    Run custom installation of VCS dependencies.

    Run installation of requirements from version control systems (VCS)
    as supported py pip.
    https://pip.pypa.io/en/stable/reference/pip_install/#vcs-support
    """

    description = "Install dependencies from VCS."

    def initialize_options(self):
        """Set initialize options."""
        pass

    def finalize_options(self):
        """Set finalize options."""
        pass

    def run(self):
        """Run installation of requirements from VCS."""
        print("setup.py sys.argv are:\n", sys.argv)
        if "--no-deps" in sys.argv:
            print("dependencies not installed")
        else:
            for package in VCS_REQUIREMENTS:
                install(package)


class build(_build):
    """Build command."""

    sub_commands = [("install_dependencies_from_vcs", None)] + _build.sub_commands


class bdist_egg(_bdist_egg):
    """Build bdist_egg."""

    def run(self):
        """Run build bdist_egg."""
        self.run_command("install_dependencies_from_vcs")
        _bdist_egg.run(self)


class develop(_develop):
    """Build develop."""

    def run(self):
        """Run build develop."""
        install_dependencies_from_vcs = self.distribution.get_command_obj(
            "install_dependencies_from_vcs"
        )
        install_dependencies_from_vcs.develop = True
        self.run_command("install_dependencies_from_vcs")
        _develop.run(self)


scripts = []

# TODO: Update these values according to the name of the module.
setup(
    name="histocartography",
    version="0.2.1",
    description="Installable histocartography package.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/histocartography/histocartography",
    download_url="https://github.com/histocartography/histocartography/archive/refs/tags/v0.2.1.tar.gz",
    author="Guillaume Jaume, Pushpak Pati, Antonio Foncubierta Rodríguez",
    author_email="guillaume.jaume2@gmail.com, pushpak.nitrkl@gmail.com, antonio.foncubierta@gmail.com",
    packages=find_packages("."),
    zip_safe=False,
    scripts=scripts,
    install_requires=PYPI_REQUIREMENTS,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
    cmdclass={
        "bdist_egg": bdist_egg,
        "build": build,
        "install_dependencies_from_vcs": install_dependencies_from_vcs,
        "develop": develop,
    },
)
