from setuptools import Extension
from setuptools import setup

root = 'cpp/src/'
src = [root + x for x in ['NPWordBeamSearch.cpp', 'WordBeamSearch.cpp', 'PrefixTree.cpp', 'LanguageModel.cpp', 'Beam.cpp']]
inc = ['cpp/src/pybind/']

word_beam_search_ext = Extension('word_beam_search', sources=src, include_dirs=inc, language='c++')
setup(name='word-beam-search', version='1.0.0', python_requires='>=3.5.0', install_requires=[], ext_modules=[word_beam_search_ext], include_package_data=False)
