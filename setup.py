import os
import platform
import shutil
import subprocess
from contextlib import contextmanager
from pathlib import Path

from skbuild import setup

package_dir = "src"
package = "pure_detector"

install_requires = ["numpy"]

root = Path(__file__).parent

try:
    git_hash = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], encoding="utf-8"
    ).strip()
except Exception:
    git_hash = ""

with open(root / package_dir / package / "_meta.py", "w") as f:
    f.writelines([f'git_hash = "{git_hash}"', ""])


cmake_args = []
if platform.system() == "Windows":
    # The Ninja cmake generator will use mingw (gcc) on windows travis instances, but we
    # need to use msvc for compatibility.
    cmake_args.append("-GVisual Studio 15 2017 Win64")
elif platform.system() == "Linux":
    # This is required for the build on manylinux.
    cmake_args.append('-DCMAKE_CXX_FLAGS="-fPIC"')
elif platform.system() == "Darwin":
    # Assuming we are using clang, we need to enable cpp17 specifically. LLVM < 5 does
    # not support -std=c++17, but we are using LLVM 4.0 on travis, so we need to use
    # c++1z. See https://clang.llvm.org/cxx_status.html
    cmake_args.append('-DCMAKE_CXX_FLAGS="-std=c++1z"')
    cmake_args.append('-DCMAKE_INSTALL_RPATH="@loader_path"')


external_package_data = []
opencv_lib_path = os.environ.get("PURE_WHEEL_OPENCV_LIB_PATH", None)
if opencv_lib_path:
    lib_dir = Path(opencv_lib_path)
    if lib_dir.exists():
        for entry in lib_dir.iterdir():
            if entry.is_file() and entry.suffix in (".dll", ".so", ".dylib"):
                external_package_data.append(entry)


@contextmanager
def collect_package_data(external):
    target = root / package_dir / package

    collected = []
    for file_path in external:
        shutil.copy(file_path, target)
        collected.append(file_path.name)
        if not (target / file_path.name).exists():
            raise FileNotFoundError(f"Could not copy data file: {file_path.name}")

    yield collected

    for file_name in collected:
        (target / file_name).unlink()


with collect_package_data(external_package_data) as package_data:
    setup(
        author="Pupil Labs",
        author_email="info@pupil-labs.com",
        cmake_args=cmake_args,
        install_requires=install_requires,
        license="GNU",
        name="pure_detector",
        packages=[package],
        package_data={package: package_data},
        package_dir={"": package_dir},
        version="0.2",
    )
