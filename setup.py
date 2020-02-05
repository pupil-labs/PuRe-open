from skbuild import setup
import subprocess


package_dir = "src"
package = "pure_detector"

install_requires = [
    "numpy",
]

try:
    git_hash = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], encoding="utf-8"
    ).strip()
except Exception:
    git_hash = ""

with open(f"{package_dir}/{package}/_meta.py", "w") as f:
    f.writelines([f'git_hash = "{git_hash}"', ""])


setup(
    author="Pupil Labs",
    author_email="info@pupil-labs.com",
    install_requires=install_requires,
    license="GNU",
    name="pure_detector",
    packages=[package],
    package_dir={"": package_dir},
    version="0.2",
)
