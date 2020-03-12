#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup
from pathlib import Path

Path("data/conf").mkdir(parents=True)
Path("data/image").mkdir(parents=True)


requirements = [
    'pyautogui',
    'PyQt5'
]
test_requirements = [
    'pyautogui',
    'PyQt5'
]

setup(
    name='labelImg',
    version='1.3.4',
    description="LabelImg is a graphical image annotation tool and label object bounding boxes in images",
    author="TzuTa Lin",
    author_email='tzu.ta.lin@gmail.com',
    url='https://github.com/vanderschuea/roLabelImg',
    packages=[
        'labelImg', 'labelImg.libs'
    ],
    package_dir={'labelImg': '.'},
    entry_points={
        'console_scripts': [
            'labelImg=labelImg.labelImg:main'
        ]
    },
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='labelImg',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
