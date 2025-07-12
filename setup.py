from setuptools import setup 

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="comind",  
    version="0.1.0",
    description="CoMind: Community-Driven MLE Agent",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="CoMind",
    packages=["comind", "competition"],
    entry_points={
        "console_scripts": [
            "comind-community=launch_community:main",
            "comind-agent=launch_agent:main",
        ]
    },
    install_requires=requirements,
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3.10",
    ]
)
