import json
import os
import shutil
import tempfile
from unittest.mock import patch

import pytest
# from mcp.client.stdio import StdioServerParameters # No longer directly asserting this type
from mcp_proxy.config_loader import NamedServerConfig, load_named_server_configs_from_file

@pytest.fixture
def create_temp_config_file():
    """Creates a temporary JSON config file and returns its path."""
    temp_files = []

    def _create_temp_config_file(config_content):
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".json"
        ) as tmp_config:
            json.dump(config_content, tmp_config)
            temp_files.append(tmp_config.name)
            return tmp_config.name

    yield _create_temp_config_file

    for f_path in temp_files:
        if os.path.exists(f_path):
            os.remove(f_path)

def test_load_valid_config(create_temp_config_file):
    config_content = {
        "mcpServers": {
            "server1": {
                "command": "echo",
                "args": ["hello"],
                "disabled": False,
                "env": {"S1_VAR": "S1_VAL"},
                "cwd": "/tmp/s1",
                "stateless": True,
            },
            "server2": {
                "command": "cat",
                "args": ["file.txt"],
                # No env, cwd, stateless specified for server2
            },
        }
    }
    tmp_config_path = create_temp_config_file(config_content)

    loaded_configs = load_named_server_configs_from_file(tmp_config_path)

    assert "server1" in loaded_configs
    s1_config = loaded_configs["server1"]
    assert isinstance(s1_config, NamedServerConfig)
    assert s1_config.command == "echo"
    assert s1_config.args == ["hello"]
    assert s1_config.disabled is False
    assert s1_config.env == {"S1_VAR": "S1_VAL"}
    assert s1_config.cwd == "/tmp/s1"
    assert s1_config.stateless is True

    assert "server2" in loaded_configs
    s2_config = loaded_configs["server2"]
    assert isinstance(s2_config, NamedServerConfig)
    assert s2_config.command == "cat"
    assert s2_config.args == ["file.txt"]
    assert s2_config.disabled is False # Default
    assert s2_config.env is None # Default
    assert s2_config.cwd is None # Default
    assert s2_config.stateless is None # Default

def test_load_config_with_disabled_server(create_temp_config_file):
    config_content = {
        "mcpServers": {
            "enabled_server": {"command": "true"},
            "disabled_server": {"command": "false", "disabled": True},
        }
    }
    tmp_config_path = create_temp_config_file(config_content)
    loaded_configs = load_named_server_configs_from_file(tmp_config_path)

    assert "enabled_server" in loaded_configs
    assert loaded_configs["enabled_server"].disabled is False

    assert "disabled_server" in loaded_configs
    assert loaded_configs["disabled_server"].disabled is True
    assert loaded_configs["disabled_server"].command == "disabled" # Placeholder

def test_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_named_server_configs_from_file("non_existent_file.json")

def test_json_decode_error(create_temp_config_file):
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".json"
    ) as tmp_config:
        tmp_config.write("this is not json {")
        tmp_config_path = tmp_config.name
    try:
        with pytest.raises(json.JSONDecodeError):
            load_named_server_configs_from_file(tmp_config_path)
    finally:
        if os.path.exists(tmp_config_path):
            os.remove(tmp_config_path)

def test_load_example_fetch_config_if_uvx_exists():
    if not shutil.which("uvx"):
        pytest.skip("uvx command not found in PATH, skipping test for example config.")

    example_config_path = os.path.join(
        os.path.dirname(__file__), "..", "config_example.json"
    )
    if not os.path.exists(example_config_path):
        pytest.fail(
            f"Example config file not found at expected path: {example_config_path}"
        )

    loaded_configs = load_named_server_configs_from_file(example_config_path)

    assert "fetch" in loaded_configs
    fetch_config = loaded_configs["fetch"]
    assert isinstance(fetch_config, NamedServerConfig)
    assert fetch_config.command == "uvx"
    assert fetch_config.args == ["mcp-server-fetch"]
    # env, cwd, stateless would be None unless specified in config_example.json for "fetch"

def test_invalid_config_format_missing_mcpServers(create_temp_config_file):
    config_content = {"some_other_key": "value"}
    tmp_config_path = create_temp_config_file(config_content)

    with pytest.raises(ValueError, match="Missing 'mcpServers' key"):
        load_named_server_configs_from_file(tmp_config_path)

@patch("mcp_proxy.config_loader.logger")
def test_invalid_server_entry_not_dict(mock_logger, create_temp_config_file):
    config_content = {"mcpServers": {"server1": "not_a_dict"}}
    tmp_config_path = create_temp_config_file(config_content)

    loaded_configs = load_named_server_configs_from_file(tmp_config_path)
    assert len(loaded_configs) == 0
    mock_logger.warning.assert_called_with(
        f"Skipping invalid server config for 'server1' in {tmp_config_path}. Entry is not a dictionary."
    )

@patch("mcp_proxy.config_loader.logger")
def test_server_entry_missing_command(mock_logger, create_temp_config_file):
    config_content = {"mcpServers": {"server_no_command": {"args": ["arg1"]}}}
    tmp_config_path = create_temp_config_file(config_content)
    loaded_configs = load_named_server_configs_from_file(tmp_config_path)
    assert "server_no_command" not in loaded_configs # It's skipped
    mock_logger.warning.assert_called_with(
        f"Named server 'server_no_command' from config is missing 'command' or it's not a string. Skipping."
    )

@patch("mcp_proxy.config_loader.logger")
def test_server_entry_invalid_args_type(mock_logger, create_temp_config_file):
    config_content = {
        "mcpServers": {
            "server_invalid_args": {"command": "mycmd", "args": "not_a_list"}
        }
    }
    tmp_config_path = create_temp_config_file(config_content)
    loaded_configs = load_named_server_configs_from_file(tmp_config_path)
    # The server is still loaded, but args defaults to []
    assert "server_invalid_args" in loaded_configs
    assert loaded_configs["server_invalid_args"].args == []
    mock_logger.warning.assert_called_with(
        f"Named server 'server_invalid_args' from config has invalid 'args' (must be a list). Defaulting to empty list."
    )

def test_empty_mcpServers_dict(create_temp_config_file):
    config_content = {"mcpServers": {}}
    tmp_config_path = create_temp_config_file(config_content)
    loaded_configs = load_named_server_configs_from_file(tmp_config_path)
    assert len(loaded_configs) == 0

def test_config_file_is_empty_json_object(create_temp_config_file):
    config_content = {}
    tmp_config_path = create_temp_config_file(config_content)
    with pytest.raises(ValueError, match="Missing 'mcpServers' key"):
        load_named_server_configs_from_file(tmp_config_path)

def test_config_file_is_empty_string(create_temp_config_file):
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".json"
    ) as tmp_config:
        tmp_config.write("")
        tmp_config_path = tmp_config.name
    try:
        with pytest.raises(json.JSONDecodeError):
            load_named_server_configs_from_file(tmp_config_path)
    finally:
        if os.path.exists(tmp_config_path):
            os.remove(tmp_config_path)

@patch("mcp_proxy.config_loader.logger")
def test_server_entry_invalid_cwd_type(mock_logger, create_temp_config_file):
    config_content = {
        "mcpServers": {
            "server_invalid_cwd": {"command": "mycmd", "cwd": 123} # cwd is not a string
        }
    }
    tmp_config_path = create_temp_config_file(config_content)
    loaded_configs = load_named_server_configs_from_file(tmp_config_path)
    assert "server_invalid_cwd" in loaded_configs
    assert loaded_configs["server_invalid_cwd"].cwd is None # Should be ignored
    mock_logger.warning.assert_called_with(
        f"Named server 'server_invalid_cwd' from config has invalid 'cwd' (must be a string). Ignoring cwd."
    )

@patch("mcp_proxy.config_loader.logger")
def test_server_entry_invalid_env_type(mock_logger, create_temp_config_file):
    config_content = {
        "mcpServers": {
            "server_invalid_env": {"command": "mycmd", "env": "not_a_dict"}
        }
    }
    tmp_config_path = create_temp_config_file(config_content)
    loaded_configs = load_named_server_configs_from_file(tmp_config_path)
    assert "server_invalid_env" in loaded_configs
    assert loaded_configs["server_invalid_env"].env is None # Should be ignored
    mock_logger.warning.assert_called_with(
        f"Named server 'server_invalid_env' from config has invalid 'env' (must be a dictionary). Ignoring env."
    )

@patch("mcp_proxy.config_loader.logger")
def test_server_entry_invalid_stateless_type(mock_logger, create_temp_config_file):
    config_content = {
        "mcpServers": {
            "server_invalid_stateless": {"command": "mycmd", "stateless": "not_a_bool"}
        }
    }
    tmp_config_path = create_temp_config_file(config_content)
    loaded_configs = load_named_server_configs_from_file(tmp_config_path)
    assert "server_invalid_stateless" in loaded_configs
    assert loaded_configs["server_invalid_stateless"].stateless is None # Should be ignored
    mock_logger.warning.assert_called_with(
        f"Named server 'server_invalid_stateless' from config has invalid 'stateless' (must be a boolean). Ignoring stateless."
    )

@patch("mcp_proxy.config_loader.logger")
def test_server_entry_invalid_disabled_type(mock_logger, create_temp_config_file):
    config_content = {
        "mcpServers": {
            "server_invalid_disabled": {"command": "mycmd", "disabled": "not_a_bool"}
        }
    }
    tmp_config_path = create_temp_config_file(config_content)
    loaded_configs = load_named_server_configs_from_file(tmp_config_path)
    assert "server_invalid_disabled" in loaded_configs
    assert loaded_configs["server_invalid_disabled"].disabled is False # Defaults to False
    mock_logger.warning.assert_called_with(
        f"Named server 'server_invalid_disabled' has non-boolean 'disabled' value. Assuming false."
    )
