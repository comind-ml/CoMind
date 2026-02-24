from setuptools import setup, find_packages

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name="comind",
    version="0.1.0",
    packages=["comind"],
    install_requires=requirements,
    python_requires=">=3.8",
    description="CoMind: Towards Community-Driven Agents for Machine Learning Engineering",
    author="CoMind",
    author_email="planarg@stu.pku.edu.cn",
)
