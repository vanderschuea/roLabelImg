#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup


requirements = [
    'pyautogui',
    'numba',
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
        'labelImg', 'labelImg.labelimg'
    ],
    package_dir={'labelImg': '.'},
    entry_points={
        'console_scripts': [
            'labelImg=labelImg:main'
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
