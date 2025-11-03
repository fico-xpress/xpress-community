"""
Setup file to ensure the proper packaging of the file to be used in Jupyter Notebook

Copyright (c) 2025 Fair Isaac Corporation
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0.
Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
"""
from setuptools import setup, find_packages
from pathlib import Path

HERE = Path(__file__).parent
README = (HERE / "README.md").read_text(encoding="utf-8") if (HERE / "README.md").exists() else ""

setup(
    name="simulator",
    version="0.1.0",
    description="Stochastic Demand Assignment - CFA simulator",
    long_description=README or "",
    long_description_content_type="text/markdown",
    author="",
    packages=find_packages(),
    package_dir={"": "."},
    include_package_data=True,
    python_requires=">=3.11",
    install_requires=[
        "pandas",
        "matplotlib",
        "xpress>=9.7"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)