"""
Import shims for the GLoRIA / MedKLIP code bases.

Both packages import their whole training stack from __init__ (augmentation
and segmentation libraries) that the grounding path never touches and that no
longer installs cleanly next to recent torch. Registering placeholder modules
before the first import satisfies those imports; if a placeholder is ever
actually used, it raises instead of silently computing something.
"""
import sys
import types


class _Missing:
    def __init__(self, *a, **k):
        raise RuntimeError("a training-only dependency was stubbed out but is being used")

    def __getattr__(self, name):
        raise AttributeError(name)


def _stub(name, attrs=()):
    if name in sys.modules:
        return
    m = types.ModuleType(name)
    # torch inspects __file__ of every module on the import stack
    m.__file__ = f"<stub:{name}>"
    m.__version__ = "0.0.0-stub"
    for a in attrs:
        setattr(m, a, _Missing)

    def _getattr(n):
        if n.startswith("__") and n.endswith("__"):
            raise AttributeError(n)
        return _Missing

    m.__getattr__ = _getattr
    sys.modules[name] = m


def install():
    _stub("segmentation_models_pytorch", ["Unet"])
    _stub("albumentations", ["ShiftScaleRotate", "Normalize", "Resize", "Compose",
                             "HorizontalFlip", "RandomBrightnessContrast",
                             "ColorJitter", "Affine"])
    _stub("albumentations.pytorch", ["ToTensorV2"])
