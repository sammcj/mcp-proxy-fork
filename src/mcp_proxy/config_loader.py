import json
import logging
import sys
import typing as t
from dataclasses import dataclass, field
from pathlib import Path # For potential CWD typing

# from mcp.client.stdio import StdioServerParameters # No longer directly returning this

logger = logging.getLogger(__name__)

@dataclass
class NamedServerConfig:
    """Configuration for a single named stdio server."""
    command: str
    args: list[str] = field(default_factory=list)
    cwd: str | Path | None = None # Working directory for the server process
    env: dict[str, str] | None = None # Environment variables for the server process
    disabled: bool = False
    # transportType: str = "stdio" # Implicitly stdio for now
    # timeout: int | None = None # Ignored for now
    stateless: bool | None = None # Per-server stateless override

def load_named_server_configs_from_file(
    config_file_path: str | Path,
    # base_env: dict[str, str] # base_env will be handled by NamedServerManager
) -> dict[str, NamedServerConfig]:
    """
    Loads named server configurations from a JSON file.

    Args:
        config_file_path: Path to the JSON configuration file.

    Returns:
        A dictionary of named server configurations.

    Raises:
        FileNotFoundError: If the config file is not found.
        json.JSONDecodeError: If the config file contains invalid JSON.
        ValueError: If the config file format is invalid.
    """
    named_server_configs: dict[str, NamedServerConfig] = {}
    logger.info(f"Loading named server configurations from: {config_file_path}")

    try:
        with open(config_file_path, "r") as f:
            config_data = json.load(f)
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_file_path}")
        raise
    except json.JSONDecodeError as e: # Specify the exception for clarity
        logger.error(f"Error decoding JSON from configuration file: {config_file_path} - {e}")
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error opening or reading configuration file {config_file_path}: {e}"
        )
        raise ValueError(f"Could not read configuration file: {e}")

    if not isinstance(config_data, dict) or "mcpServers" not in config_data:
        msg = f"Invalid config file format in {config_file_path}. Missing 'mcpServers' key."
        logger.error(msg)
        raise ValueError(msg)

    for name, server_data in config_data.get("mcpServers", {}).items():
        if not isinstance(server_data, dict):
            logger.warning(
                f"Skipping invalid server config for '{name}' in {config_file_path}. Entry is not a dictionary."
            )
            continue

        is_disabled = server_data.get("disabled", False)
        if not isinstance(is_disabled, bool):
            logger.warning(f"Named server '{name}' has non-boolean 'disabled' value. Assuming false.")
            is_disabled = False
        if is_disabled:
            logger.info(f"Named server '{name}' from config is disabled. Skipping.")
            named_server_configs[name] = NamedServerConfig(
                command="disabled", # Placeholder command
                disabled=True
            ) # Still add it so manager knows it's explicitly disabled
            continue

        command = server_data.get("command")
        args = server_data.get("args", [])
        cwd = server_data.get("cwd")
        env = server_data.get("env")
        stateless = server_data.get("stateless")


        if not command or not isinstance(command, str):
            logger.warning(
                f"Named server '{name}' from config is missing 'command' or it's not a string. Skipping."
            )
            continue
        if not isinstance(args, list):
            logger.warning(
                f"Named server '{name}' from config has invalid 'args' (must be a list). Defaulting to empty list."
            )
            args = []
        if cwd is not None and not isinstance(cwd, str):
            logger.warning(
                f"Named server '{name}' from config has invalid 'cwd' (must be a string). Ignoring cwd."
            )
            cwd = None
        if env is not None and not isinstance(env, dict):
            logger.warning(
                f"Named server '{name}' from config has invalid 'env' (must be a dictionary). Ignoring env."
            )
            env = None
        if stateless is not None and not isinstance(stateless, bool):
            logger.warning(
                f"Named server '{name}' from config has invalid 'stateless' (must be a boolean). Ignoring stateless."
            )
            stateless = None


        named_server_configs[name] = NamedServerConfig(
            command=command,
            args=args,
            cwd=cwd,
            env=env,
            disabled=False, # Already handled above, but for clarity
            stateless=stateless,
        )
        logger.info(
            f"Parsed named server '{name}' from config: {command} {' '.join(args)}"
        )

    return named_server_configs

def parse_named_server_config_string(name: str, command_string: str) -> NamedServerConfig | None:
    """
    Parses a command string into a NamedServerConfig.
    This is primarily for handling --named-server CLI arguments.
    """
    try:
        parts = shlex.split(command_string)
        if not parts:
            logger.error(f"Empty command string for named server '{name}'.")
            return None
        command = parts[0]
        args = parts[1:]
        # CLI-defined servers get default cwd (None), env (None), stateless (None)
        return NamedServerConfig(command=command, args=args)
    except Exception as e:
        logger.error(f"Error parsing command string for named server '{name}': '{command_string}' - {e}")
        return None

# Example of how StdioServerParameters was used, for reference if needed later
# from mcp.client.stdio import StdioServerParameters
# StdioServerParameters(
# command=command,
# args=command_args,
# env=base_env.copy(), # Each server gets a copy of base_env
# cwd=None, # Named servers from config currently run in proxy's CWD
# )
