from setuptools import find_packages, setup
import sys

package_name = 'vlm_1'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    options={
        'build_scripts': {
            'executable': sys.executable,
        },
    },

    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='wang',
    maintainer_email='wang-2005@users.noreply.github.com',
    description='ROS 2 YOLO and Qwen2-VL robot vision prototype',
    license='All rights reserved',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        'yolo_node = vlm_1.yolo_node:main',
        'roi_node = vlm_1.ROI_node:main',
        'vlm_node = vlm_1.vlm_node:main',
        ],
    },
)
