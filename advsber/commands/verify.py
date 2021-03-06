"""Taken from https://github.com/allenai/allennlp/blob/master/scripts/verify.py."""

import argparse
import sys
from subprocess import CalledProcessError, run


def main(checks):
    try:
        print("Verifying with " + str(checks))
        if "pytest" in checks:
            print("Tests (pytest):", flush=True)
            run("pytest --color=yes -rf", shell=True, check=True)

        if "flake8" in checks:
            print("Linter (flake8)", flush=True)
            run("flake8 -v", shell=True, check=True)
            print("flake8 checks passed")

        if "black" in checks:
            print("Formatter (black)", flush=True)
            run("black -v --check .", shell=True, check=True)
            print("black checks passed")

        if "mypy" in checks:
            print("Typechecker (mypy):", flush=True)
            run(
                "mypy advsber"
                # This is necessary because not all the imported libraries have type stubs.
                " --ignore-missing-imports"
                # This is necessary because PyTorch has some type stubs but they're incomplete,
                # and mypy will follow them and generate a lot of spurious errors.
                " --no-site-packages"
                # We are extremely lax about specifying Optional[] types, so we need this flag.
                # TODO: tighten up our type annotations and remove this
                " --no-strict-optional"
                # Some versions of mypy crash randomly when caching, probably because of our use of
                # NamedTuple (https://github.com/python/mypy/issues/7281).
                " --cache-dir=/dev/null",
                shell=True,
                check=True,
            )
            print("mypy checks passed")

    except CalledProcessError:
        # squelch the exception stacktrace
        sys.exit(1)


if __name__ == "__main__":
    checks = [
        "pytest",
        "flake8",
        "mypy",
        "black",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--checks", default=checks, nargs="+", choices=checks)

    args = parser.parse_args()

    main(args.checks)
