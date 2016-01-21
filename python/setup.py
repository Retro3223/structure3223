from distutils.core import setup, Extension

setup(
    name='structure3223',
    version='1.0',
    description='wrap some structure sensor reading routines',
    ext_modules=[
        Extension(
            'structure3223',
            sources=['structure3223.cpp'],
            libraries=['OpenNI2']
        )
    ]
)
