"""
The :mod:`deslib.static` provides a set of static ensemble methods which are
often used as a baseline to compare the performance of dynamic selection
algorithms.
"""

from .oracle import Oracle
from .single_best import SingleBest
from .static_selection import StaticSelection
from .stacked import StackedClassifier
from .epfh import EnsemblePruneFH

__all__ = ['Oracle',
           'SingleBest',
           'StaticSelection',
           'StackedClassifier',
           'EnsemblePruneFH']
