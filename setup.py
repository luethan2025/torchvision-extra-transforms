import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name="torchvision-extra-transforms",
        version="0.1.0",
        description=" ",
        install_requires=[
            "pillow",
            "torch",
            "torchvision",
        ],
        packages=setuptools.find_namespace_packages(where=".", include=["torchvision*"]),
        package_dir={"torchvision": "torchvision"},
        python_requires=">=3.10",
    )
