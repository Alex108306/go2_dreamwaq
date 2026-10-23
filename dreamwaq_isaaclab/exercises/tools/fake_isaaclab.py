"""Minimal stub for importing project modules without Isaac Sim.

Files like `rewards.py` import `isaaclab.*`, but the computation the exercises check is
pure torch. Launching Isaac Sim takes 25 seconds, so we plant fake modules that only let the import
pass into `sys.modules` and check in 1 second.

Names are not enumerated one by one — any attribute access returns a dummy class, so
even if the project starts importing new symbols, the checkers don't need to change.

    from fake_isaaclab import install_stubs, load_module

    install_stubs()
    rewards = load_module(Path(".../rewards.py"))
"""
from __future__ import annotations

import importlib.abc
import importlib.util
import sys
import types
from pathlib import Path

# Every module starting with these prefixes is faked. Since we intercept with a meta path finder rather
# than enumerating names, nothing here needs to change when the project starts importing a new isaaclab submodule.
_STUB_PREFIXES = ("isaaclab", "isaacsim", "omni", "carb", "pxr", "usdrt")


class _DummyMeta(type):
    """Also handles class-level attribute access (e.g. RayCasterCfg.OffsetCfg)."""

    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        sub = _DummyMeta(name, (_Dummy,), {})
        setattr(cls, name, sub)
        return sub

    def __call__(cls, *args, **kwargs):
        return super().__call__(*args, **kwargs)


class _Dummy(metaclass=_DummyMeta):
    """A placeholder that never dies whatever it is used as (base class, type hint, decorator)."""

    def __init__(self, *args, **kwargs):
        # Also used as a base class that takes (cfg, env), like ManagerTermBase.
        self.cfg = kwargs.get("cfg", args[0] if args else None)
        self._env = kwargs.get("env", args[1] if len(args) > 1 else None)

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        return _Dummy()

    @classmethod
    def __class_getitem__(cls, item):
        return cls


class _AnyModule(types.ModuleType):
    """A module that returns a dummy for any attribute requested. It has __path__ so it also works as a package."""

    def __init__(self, name: str):
        super().__init__(name)
        self.__path__ = []  # allow submodule imports

    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        # Names starting with an uppercase letter are treated as classes, otherwise as functions/values.
        value = _Dummy if name[:1].isupper() else _Dummy()
        setattr(self, name, value)
        return value


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Creates a fake module on the fly when `isaaclab.*` etc. is requested."""

    def find_spec(self, fullname, path=None, target=None):
        root = fullname.split(".")[0]
        # Prefix matching — also catches isaaclab_physx, isaaclab_tasks, isaaclab_assets, etc.
        if not any(root.startswith(pfx) for pfx in _STUB_PREFIXES):
            return None
        return importlib.util.spec_from_loader(fullname, self, is_package=True)

    def create_module(self, spec):
        return _AnyModule(spec.name)

    def exec_module(self, module):
        pass


_FINDER = _StubFinder()


def install_stubs() -> None:
    """Install the `isaaclab.*` stub loader into sys.meta_path (without installing it twice)."""
    if not any(isinstance(f, _StubFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _FINDER)
    # If the real one is already loaded the stub would be shadowed, so remove it.
    for name in list(sys.modules):
        root = name.split(".")[0]
        if any(root.startswith(p) for p in _STUB_PREFIXES) and not isinstance(sys.modules[name], _AnyModule):
            del sys.modules[name]


def load_module(path: Path, alias: str | None = None, extra_paths: tuple[Path, ...] = ()):
    """Import a .py from an arbitrary path (without touching sys.path).

    So that files using relative imports like `from .sibling import X` can be loaded too,
    a fake parent package whose __path__ is the file's directory is created and attached.

    A starter is generated alone without its sibling modules, so if you pass the original source directory as
    `extra_paths`, siblings are looked up there as well.
    """
    if not path.exists():
        raise SystemExit(
            f"File not found: {path}\n"
            f"  generate the starters with exercises/tools/make_exercise.py --all"
        )
    install_stubs()

    pkg_name = f"_expkg_{abs(hash(str(path.parent))) & 0xFFFFFFFF:x}"
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(path.parent), *(str(x) for x in extra_paths)]
        sys.modules[pkg_name] = pkg

    name = alias or f"{pkg_name}.{path.stem}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = pkg_name
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
