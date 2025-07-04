from setuptools import setup, find_packages

setup(
    name="open_cyber_glove",
    version="0.1.0",
    description="Open-source Python SDK for interfacing with data gloves, supporting real-time sensor data acquisition, calibration, and extensible inference.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="OpenCyberGlove Contributors",
    license="BSD 3-Clause",
    packages=find_packages(),
    install_requires=[
        "pyserial",
        "numpy",
        "matplotlib",
        "tqdm",
        "open3d",
        "onnxruntime"
    ],
    python_requires=">=3.7",
    url="https://github.com/CyberOrigin2077/open-cyber-glove",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
) 