import os
# import shutil
import sys
from pathlib import Path
from typing import (Any, Dict, Mapping, MutableMapping, Optional, Tuple, Union,
                    cast)

import toml
import trafaret as t
from toml.decoder import InlineTableDict

__all__ = (
    "ConfigurationError",
    "read_from_file",
    "override_key",
    "override_with_env",
    "check",
    "merge",
)


class ConfigurationError(Exception):

    invalid_data: Mapping[str, Any]

    def __init__(self, invalid_data: Mapping[str, Any]) -> None:
        super().__init__(invalid_data)
        self.invalid_data = invalid_data


def find_config_file(daemon_name: str) -> Path:
    toml_path_from_env = os.environ.get("BACKEND_CONFIG_FILE", None)
    if not toml_path_from_env:
        toml_paths = [
            Path.cwd() / f"{daemon_name}.toml",
        ]
        if sys.platform.startswith("linux") or sys.platform.startswith("darwin"):
            toml_paths += [
                Path.home() / ".config" / "backend.ai" / f"{daemon_name}.toml",
                Path(f"/etc/backend.ai/{daemon_name}.toml"),
            ]
        else:
            raise ConfigurationError(
                {
                    "read_from_file()": f"Unsupported platform for config path auto-discovery: {sys.platform}",
                }
            )
    else:
        toml_paths = [Path(toml_path_from_env)]
    for _path in toml_paths:
        if _path.is_file():
            return _path
    else:
        searched_paths = ",".join(map(str, toml_paths))
        raise ConfigurationError(
            {
                "find_config_file()": f"Could not read config from: {searched_paths}",
            }
        )


def read_from_file(
    toml_path: Optional[Union[Path, str]], daemon_name: str
) -> Tuple[Dict[str, Any], Path]:
    config: Dict[str, Any]
    discovered_path: Path
    if toml_path is None:
        discovered_path = find_config_file(daemon_name)
        # copy_config_file_to_frontend(daemon_name)
    else:
        discovered_path = Path(toml_path)
    try:
        config = cast(Dict[str, Any], toml.loads(discovered_path.read_text()))
        config = _sanitize_inline_dicts(config)
    except IOError:
        raise ConfigurationError(
            {
                "read_from_file()": f"Could not read config from: {discovered_path}",
            }
        )
    else:
        return config, discovered_path


# def copy_config_file_to_frontend(daemon_name: str):
#     source = find_config_file(daemon_name)
#     destination = str(Path.cwd()) + f"/frontend/configs/{daemon_name}.toml"
#     directory = str(Path.cwd()) + "/frontend/configs"
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     if os.path.isfile(destination):
#         pass
#     else:
#         shutil.copyfile(source, destination)


def override_key(
    table: MutableMapping[str, Any], key_path: Tuple[str, ...], value: Any
):
    for k in key_path[:-1]:
        if k not in table:
            table[k] = {}
        table = table[k]
    table[key_path[-1]] = value


def override_with_env(
    table: MutableMapping[str, Any], key_path: Tuple[str, ...], env_key: str
):
    val = os.environ.get(env_key, None)
    if val is None:
        return
    override_key(table, key_path, val)


def check(table: Any, iv: t.Trafaret):
    try:
        config = iv.check(table)
    except t.DataError as e:
        raise ConfigurationError(e.as_dict())
    else:
        return config


def merge(table: Mapping[str, Any], updates: Mapping[str, Any]) -> Mapping[str, Any]:
    result = {**table}
    for k, v in updates.items():
        if isinstance(v, Mapping):
            orig = result.get(k, {})
            assert isinstance(orig, Mapping)
            result[k] = merge(orig, v)
        else:
            result[k] = v
    return result


def _sanitize_inline_dicts(table: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for k, v in table.items():
        if isinstance(v, InlineTableDict):
            # Since this function always returns a copied dict,
            # this automatically converts InlineTableDict to dict.
            result[k] = _sanitize_inline_dicts(v)
        elif isinstance(v, Dict):
            result[k] = _sanitize_inline_dicts(v)
        else:
            result[k] = v
    return result
