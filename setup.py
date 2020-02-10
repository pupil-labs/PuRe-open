import platform
import subprocess

from skbuild import setup

package_dir = "src"
package = "pure_detector"

install_requires = ["numpy", "opencv-python"]

try:
    git_hash = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], encoding="utf-8"
    ).strip()
except Exception:
    git_hash = ""

with open(f"{package_dir}/{package}/_meta.py", "w") as f:
    f.writelines([f'git_hash = "{git_hash}"', ""])


cmake_args = []
if platform.system() == "Windows":
    # The Ninja cmake generator will use mingw (gcc) on windows travis instances, but we
    # need to use msvc for compatibility.
    cmake_args.append("-GVisual Studio 15 2017 Win64")


setup(
    author="Pupil Labs",
    author_email="info@pupil-labs.com",
    cmake_args=cmake_args,
    install_requires=install_requires,
    license="GNU",
    name="pure_detector",
    packages=[package],
    package_dir={"": package_dir},
    version="0.2",
)
