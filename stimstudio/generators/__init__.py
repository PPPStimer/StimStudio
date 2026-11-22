"""Signal generators and the generator registry.

Importing this package registers all built-in generators.
"""

from .base import (
    Generator,
    ParamSpec,
    get_generator,
    list_generators,
    register,
)

# Import side effects register each generator.
from . import waveforms  # noqa: F401
from . import channels  # noqa: F401
from . import activities  # noqa: F401
from . import wav_import  # noqa: F401
from . import draw  # noqa: F401
from . import adsr  # noqa: F401
from . import steps  # noqa: F401
from . import noise  # noqa: F401
from . import rhythm  # noqa: F401
from . import triphase  # noqa: F401
from . import stim  # noqa: F401
from . import perceptual  # noqa: F401
from . import complexity  # noqa: F401
from . import longform  # noqa: F401

__all__ = [
    "Generator",
    "ParamSpec",
    "get_generator",
    "list_generators",
    "register",
]
