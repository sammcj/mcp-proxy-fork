"""Create a local SSE server that proxies requests to a stdio MCP server."""

import asyncio
import contextlib
import json
import logging
import shlex
from collections.abc import AsyncIterator, Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import watchfiles # type: ignore
import uvicorn
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.server import Server as MCPServerSDK
from mcp.server.sse import SseServerTransport
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount, Route
from starlette.types import ASGIApp, Receive, Scope, Send

from .config_loader import (
    load_named_server_configs_from_file,
    parse_named_server_config_string,
    NamedServerConfig, # Import from config_loader
)
from .proxy_server import create_proxy_server

logger = logging.getLogger(__name__)

@dataclass
class MCPServerSettings:
    """Settings for the MCP server."""

    bind_host: str
    port: int
    stateless: bool = False
    allow_origins: list[str] | None = None
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

_global_status: dict[str, Any] = {
    "api_last_activity": datetime.now(timezone.utc).isoformat(),
    "server_instances": {}
}

def _update_global_activity(server_name: str = "global") -> None:
    now_iso = datetime.now(timezone.utc).isoformat()
    _global_status["api_last_activity"] = now_iso
    if server_name != "global" and server_name in _global_status["server_instances"]:
        if isinstance(_global_status["server_instances"][server_name], dict):
            _global_status["server_instances"][server_name]["last_activity"] = now_iso
        else: # old format, just update global
             _global_status["server_instances"][server_name] = f"active, last_activity: {now_iso}"


async def _handle_status(_: Request) -> Response:
    """Global health check and service usage monitoring endpoint."""
    return JSONResponse(_global_status)

# Removed local definition of NamedServerConfig, it's now imported

@dataclass
class NamedServerInstance:
    """Holds all components for a running named server instance."""
    config: NamedServerConfig # This will now use the imported NamedServerConfig
    sdk_proxy: MCPServerSDK[Any]
    http_manager: StreamableHTTPSessionManager
    sse_transport: SseServerTransport
    exit_stack: contextlib.AsyncExitStack
    task: asyncio.Task[None] | None = None # For the stdio_client task

class NamedServerManager:
    """Manages the lifecycle of named MCP server instances."""

    def __init__(self, base_env: dict[str, str] | None, stateless_default: bool):
        self._servers: dict[str, NamedServerInstance] = {}
        self._base_env = base_env or {}
        self._stateless_default = stateless_default
        self._lock = asyncio.Lock()

    async def _create_new_server_instance(self, name: str, config: NamedServerConfig) -> NamedServerInstance | None:
        logger.info(f"Creating new server instance: {name} with command '{config.command} {' '.join(config.args)}'")
        exit_stack = contextlib.AsyncExitStack()
        try:
            # Prepare StdioServerParameters
            env = self._base_env.copy()
            if config.env:
                env.update(config.env)

            params = StdioServerParameters(
                command=config.command,
                args=config.args,
                cwd=config.cwd,
                env=env,
            )

            stdio_streams = await exit_stack.enter_async_context(stdio_client(params))
            session = await exit_stack.enter_async_context(ClientSession(*stdio_streams))
            sdk_proxy = await create_proxy_server(session)

            sse_transport = SseServerTransport("/messages/") # Path is relative to the server's mount point
            stateless_for_instance = config.stateless if config.stateless is not None else self._stateless_default
            http_manager = StreamableHTTPSessionManager(
                app=sdk_proxy,
                event_store=None,
                json_response=True,
                stateless=stateless_for_instance,
            )
            await exit_stack.enter_async_context(http_manager.run())

            instance = NamedServerInstance(
                config=config,
                sdk_proxy=sdk_proxy,
                http_manager=http_manager,
                sse_transport=sse_transport,
                exit_stack=exit_stack,
            )
            _global_status["server_instances"][name] = {
                "status": "running",
                "command": f"{config.command} {' '.join(config.args)}",
                "last_activity": datetime.now(timezone.utc).isoformat(),
                "stateless": stateless_for_instance,
            }
            logger.info(f"Successfully created server instance: {name}")
            return instance
        except Exception as e:
            logger.error(f"Failed to create server instance {name}: {e}", exc_info=True)
            await exit_stack.aclose()
            _global_status["server_instances"][name] = {"status": "error", "reason": str(e)}
            return None

    async def update_servers(self, new_configs: dict[str, NamedServerConfig]) -> None:
        async with self._lock:
            logger.info(f"Updating servers. New config has {len(new_configs)} servers.")
            current_server_names = set(self._servers.keys())
            new_server_names = set(new_configs.keys())

            to_remove = current_server_names - new_server_names
            to_add = new_server_names - current_server_names
            to_check = current_server_names.intersection(new_server_names)

            for name in to_remove:
                logger.info(f"Removing server: {name}")
                instance = self._servers.pop(name, None)
                if instance:
                    await instance.exit_stack.aclose()
                _global_status["server_instances"].pop(name, None)
                logger.info(f"Server {name} removed.")

            for name in to_add:
                logger.info(f"Adding new server: {name}")
                new_instance = await self._create_new_server_instance(name, new_configs[name])
                if new_instance:
                    self._servers[name] = new_instance
                logger.info(f"Server {name} added (or failed to add).")


            for name in to_check:
                logger.info(f"Checking server for updates: {name}")
                current_instance = self._servers[name]
                new_config = new_configs[name]
                # Simple comparison: if command, args, cwd or env changed, restart.
                # More sophisticated checks could be added (e.g. deep dict comparison for env)
                if (current_instance.config.command != new_config.command or
                        current_instance.config.args != new_config.args or
                        current_instance.config.cwd != new_config.cwd or
                        current_instance.config.env != new_config.env or
                        current_instance.config.stateless != new_config.stateless): # Check stateless
                    logger.info(f"Configuration changed for server: {name}. Restarting.")
                    await current_instance.exit_stack.aclose()
                    _global_status["server_instances"].pop(name, None)
                    re_instance = await self._create_new_server_instance(name, new_config)
                    if re_instance:
                        self._servers[name] = re_instance
                    logger.info(f"Server {name} restarted (or failed to restart).")
                else:
                    logger.info(f"No changes detected for server: {name}")
            logger.info("Server update process complete.")

    def get_server_handlers(self, server_name: str) -> tuple[StreamableHTTPSessionManager, SseServerTransport, MCPServerSDK[Any]] | None:
        instance = self._servers.get(server_name)
        if instance:
            return instance.http_manager, instance.sse_transport, instance.sdk_proxy
        return None

    async def close_all(self) -> None:
        async with self._lock:
            logger.info("Closing all named servers...")
            for name, instance in list(self._servers.items()): # Iterate over a copy
                logger.info(f"Closing server: {name}")
                await instance.exit_stack.aclose()
                self._servers.pop(name, None)
                _global_status["server_instances"].pop(name, None)
            logger.info("All named servers closed.")

async def dispatch_to_named_server(request: Request) -> None:
    """Dynamically dispatches requests to the correct named server."""
    server_manager = request.app.state.named_server_manager
    path_params = request.path_params
    server_name_and_subpath = path_params.get("server_name_and_subpath", "")

    parts = server_name_and_subpath.split("/", 1)
    server_name = parts[0]
    subpath = f"/{parts[1]}" if len(parts) > 1 else "/"

    handlers = server_manager.get_server_handlers(server_name)
    if not handlers:
        logger.warning(f"Server '{server_name}' not found for path '{server_name_and_subpath}'")
        response = JSONResponse({"error": f"Server '{server_name}' not found"}, status_code=404)
        await response(request.scope, request.receive, request.send)
        return

    http_manager, sse_transport, sdk_proxy = handlers
    _update_global_activity(server_name)

    # Modify scope to reflect the subpath for the specific server's routing
    original_path = request.scope["path"]

    # The dynamic route captures /servers/{server_name_and_subpath:path}
    # We need to strip /servers/{server_name}/ from the path to get the subpath for the instance
    base_path_to_strip = f"/servers/{server_name}"

    if original_path.startswith(base_path_to_strip):
        instance_path = original_path[len(base_path_to_strip):]
        if not instance_path.startswith("/"):
            instance_path = "/" + instance_path
    else: # Should not happen with correct routing
        instance_path = subpath

    # logger.debug(f"Dispatching to server '{server_name}'. Original path: '{original_path}', Instance path: '{instance_path}'")

    request.scope["path"] = instance_path
    # Ensure root_path is also adjusted if necessary, though for sub-apps it might be fine
    # request.scope["root_path"] = request.scope["root_path"] + base_path_to_strip


    if instance_path.startswith("/sse"): # SSE endpoint for the named server
        async with sse_transport.connect_sse(
            request.scope,
            request.receive,
            request._send,  # noqa: SLF001
        ) as (read_stream, write_stream):
            await sdk_proxy.run(
                read_stream,
                write_stream,
                sdk_proxy.create_initialization_options(),
            )
    elif instance_path.startswith("/messages/"): # SSE message POST endpoint
        await sse_transport.handle_post_message(request.scope, request.receive, request.send)
    else: # Streamable HTTP endpoint for the named server (typically /mcp)
        await http_manager.handle_request(request.scope, request.receive, request.send)


async def _config_file_watcher(
    config_path: Path,
    server_manager: NamedServerManager,
    initial_load_event: asyncio.Event,
) -> None:
    """Watches the config file for changes and updates servers."""
    logger.info(f"Starting config file watcher for: {config_path}")
    # Initial load
    try:
        if config_path.exists():
            logger.info(f"Performing initial load of config file: {config_path}")
            configs = load_named_server_configs_from_file(config_path)
            await server_manager.update_servers(configs)
        else:
            logger.warning(f"Initial config file not found: {config_path}. No servers loaded initially.")
            await server_manager.update_servers({}) # Ensure any existing are cleared if file deleted
    except Exception as e:
        logger.error(f"Error during initial config load from {config_path}: {e}", exc_info=True)
    finally:
        initial_load_event.set() # Signal that initial load attempt is complete

    async for changes in watchfiles.awatch(config_path.parent if config_path else ".", stop_event=asyncio.Event()): # Watch parent dir
        for change_type, changed_path_str in changes:
            changed_path = Path(changed_path_str)
            if changed_path.resolve() == config_path.resolve(): # Only react to changes to our specific config file
                logger.info(f"Config file {config_path} changed ({change_type.name}). Reloading...")
                try:
                    if change_type == watchfiles.Change.deleted or not config_path.exists():
                        logger.info(f"Config file {config_path} deleted. Clearing all named servers.")
                        await server_manager.update_servers({})
                    else:
                        configs = load_named_server_configs_from_file(config_path)
                        await server_manager.update_servers(configs)
                    logger.info("Server configurations reloaded due to file change.")
                except json.JSONDecodeError as e:
                    logger.error(f"Error decoding JSON from {config_path}: {e}. Servers not updated.")
                except Exception as e:
                    logger.error(f"Error reloading server configurations from {config_path}: {e}", exc_info=True)
                break # Processed our file, no need to check other changes in this batch

async def run_mcp_server( # Modified signature
    mcp_settings: MCPServerSettings,
    default_server_params: StdioServerParameters | None = None,
    # named_server_params_cli: dict[str, StdioServerParameters] | None = None, # Replaced by config file logic
    named_server_config_path: Path | None = None,
    base_env: dict[str, str] | None = None,
    raw_named_server_args: list[tuple[str, str]] | None = None, # For CLI --named-server
) -> None:
    """Run stdio client(s) and expose an MCP server with multiple possible backends."""

    server_manager = NamedServerManager(base_env=base_env, stateless_default=mcp_settings.stateless)
    initial_config_load_done = asyncio.Event()
    watcher_task = None

    all_routes: list[Route | Mount] = [
        Route("/status", endpoint=_handle_status),
    ]

    async with contextlib.AsyncExitStack() as stack:
        # Lifespan for the main Starlette app
        @contextlib.asynccontextmanager
        async def app_lifespan(_app: Starlette) -> AsyncIterator[None]:
            nonlocal watcher_task
            logger.info("Main application lifespan starting...")
            _app.state.named_server_manager = server_manager # Make manager accessible

            if named_server_config_path:
                watcher_task = asyncio.create_task(
                    _config_file_watcher(named_server_config_path, server_manager, initial_config_load_done)
                )
                # Wait for the initial load to complete before server fully starts accepting requests for named servers
                await initial_config_load_done.wait()
                logger.info("Initial configuration load complete (or skipped if file not found).")
            elif raw_named_server_args: # Load from CLI args if no config file
                logger.info("Loading named servers from CLI arguments.")
                cli_configs: dict[str, NamedServerConfig] = {}
                for name, cmd_str in raw_named_server_args:
                    try:
                        parsed_cmd = shlex.split(cmd_str)
                        command = parsed_cmd[0]
                        args = parsed_cmd[1:]
                        cli_configs[name] = NamedServerConfig(command=command, args=args) # Fill with defaults
                    except Exception as e:
                        logger.error(f"Error parsing CLI named server '{name}' with command '{cmd_str}': {e}")
                await server_manager.update_servers(cli_configs)
                logger.info("CLI named servers processed.")
            else:
                logger.info("No named server config file or CLI named server args provided.")


            # Setup default server if configured (independent of named servers)
            if default_server_params:
                logger.info(f"Setting up default server: {default_server_params.command} {' '.join(default_server_params.args)}")
                try:
                    # This part needs to be adapted or use a similar mechanism as NamedServerManager
                    # For simplicity, let's assume default server is static and doesn't use NamedServerManager directly for now
                    # Or, it could be treated as a special named server by NamedServerManager if desired.
                    # The original plan didn't detail default server with NamedServerManager.
                    # For now, keep its setup separate as it was.
                    stdio_streams_default = await stack.enter_async_context(stdio_client(default_server_params))
                    session_default = await stack.enter_async_context(ClientSession(*stdio_streams_default))
                    proxy_default = await create_proxy_server(session_default)

                    # Create routes for the default server
                    sse_transport_default = SseServerTransport("/messages/")
                    http_manager_default = StreamableHTTPSessionManager(
                        app=proxy_default, event_store=None, json_response=True, stateless=mcp_settings.stateless
                    )
                    await stack.enter_async_context(http_manager_default.run())

                    async def handle_sse_default(request: Request) -> None:
                        async with sse_transport_default.connect_sse(
                            request.scope, request.receive, request._send,
                        ) as (read_stream, write_stream):
                            _update_global_activity("default")
                            await proxy_default.run(read_stream, write_stream, proxy_default.create_initialization_options())

                    async def handle_http_default(scope: Scope, receive: Receive, send: Send) -> None:
                        _update_global_activity("default")
                        await http_manager_default.handle_request(scope, receive, send)

                    default_routes = [
                        Mount("/mcp", app=handle_http_default),
                        Route("/sse", endpoint=handle_sse_default),
                        Mount("/messages/", app=sse_transport_default.handle_post_message),
                    ]
                    all_routes.extend(default_routes) # Add default routes to the global list
                    _global_status["server_instances"]["default"] = {
                        "status": "running",
                        "command": f"{default_server_params.command} {' '.join(default_server_params.args)}",
                        "last_activity": datetime.now(timezone.utc).isoformat(),
                        "stateless": mcp_settings.stateless,
                    }
                    logger.info("Default server configured and routes added.")

                except Exception as e:
                    logger.error(f"Failed to set up default server: {e}", exc_info=True)
                    _global_status["server_instances"]["default"] = {"status": "error", "reason": str(e)}


            # Add the dynamic route for named servers
            # This single route handles all /servers/* paths
            all_routes.append(Route("/servers/{server_name_and_subpath:path}", endpoint=dispatch_to_named_server))
            logger.info("Dynamic route for named servers added.")

            yield # Uvicorn server runs here

            logger.info("Main application lifespan shutting down...")
            if watcher_task:
                watcher_task.cancel()
                try:
                    await watcher_task
                except asyncio.CancelledError:
                    logger.info("Config watcher task cancelled.")
            await server_manager.close_all()
            logger.info("All server resources cleaned up.")

        if not default_server_params and not named_server_config_path and not raw_named_server_args:
            logger.error("No default server, named server config file, or CLI named server args provided. At least one server configuration type is required.")
            # Optionally, could raise an error or exit if no servers are possible.
            # For now, it will start an empty server with just /status and /servers/* (which will 404).
            # This might be desired if config file is expected to be created later.

        middleware: list[Middleware] = []
        if mcp_settings.allow_origins:
            middleware.append(
                Middleware(
                    CORSMiddleware,
                    allow_origins=mcp_settings.allow_origins,
                    allow_methods=["*"],
                    allow_headers=["*"],
                ),
            )

        starlette_app = Starlette(
            debug=(mcp_settings.log_level == "DEBUG"),
            routes=all_routes,
            middleware=middleware,
            lifespan=app_lifespan,
        )

        config = uvicorn.Config(
            starlette_app,
            host=mcp_settings.bind_host,
            port=mcp_settings.port,
            log_level=mcp_settings.log_level.lower(),
            # Disable lifespan 'startup_failure_status_code' to prevent uvicorn exiting on initial config load error
            # This allows the watcher to potentially recover if the file is fixed.
            # However, uvicorn's Config doesn't directly expose this. We rely on graceful error handling.
        )
        http_server = uvicorn.Server(config)
        logger.info(
            "Serving incoming MCP requests on %s:%s",
            mcp_settings.bind_host,
            mcp_settings.port,
        )
        await http_server.serve()
