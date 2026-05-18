from __future__ import annotations

from .arguments import parse_args


_GLOBAL_ARGS = None


def set_global_variables(
	extra_args_provider=None,
	args_defaults: dict[str, object] | None = None,
	ignore_unknown_args: bool = False,
	argv: list[str] | None = None,
):
	"""Parse and register the global runtime arguments."""
	args = parse_args(
		extra_args_provider=extra_args_provider,
		defaults=args_defaults,
		ignore_unknown_args=ignore_unknown_args,
		argv=argv,
	)
	set_args(args)
	return args


def set_args(args) -> None:
	"""Register a prebuilt args object.

	This is mainly useful for tests or small scripts that build a Namespace
	manually and still want to share it through get_args().
	"""
	global _GLOBAL_ARGS
	_ensure_var_is_not_initialized(_GLOBAL_ARGS, "args")
	_GLOBAL_ARGS = args


def get_args():
	"""Return the global args object."""
	_ensure_var_is_initialized(_GLOBAL_ARGS, "args")
	return _GLOBAL_ARGS


def destroy_global_variables() -> None:
	"""Reset global state.

	Intended for tests only.
	"""
	global _GLOBAL_ARGS
	_GLOBAL_ARGS = None


def _ensure_var_is_initialized(var, name: str) -> None:
	if var is None:
		raise RuntimeError(f"{name} is not initialized.")


def _ensure_var_is_not_initialized(var, name: str) -> None:
	if var is not None:
		raise RuntimeError(f"{name} is already initialized.")
