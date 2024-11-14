from setuptools import setup, find_packages

setup(
    name="boost_track",
    version="0.1.0",
    packages=find_packages(include=["boosttrack*"]),
    install_requires=[
        "torchreid",
         "yacs",
    ]
)
