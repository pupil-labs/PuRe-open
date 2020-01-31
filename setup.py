from skbuild import setup

package_dir = "src"
package = "pure_detector"

install_requires = [
    "numpy",
]

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