"""Create a local SSE server that proxies requests to a stdio MCP server."""

import contextlib
import asyncio # Added for asyncio.CancelledError
from typing import Any
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import uvicorn
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.server import Server as MCPServerSDK  # Renamed to avoid conflict
from mcp.server.sse import SseServerTransport
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, PlainTextResponse
from starlette.routing import Mount, Route
from starlette.types import Receive, Scope, Send
from starlette.exceptions import HTTPException

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

_global_status = {
    "api_last_activity": datetime.now(timezone.utc).isoformat(),
    "server_instances": {}
}

def _update_global_activity() -> None:
    _global_status["api_last_activity"] = datetime.now(timezone.utc).isoformat()

async def _handle_status(_: Request) -> Response:
    """Global health check and service usage monitoring endpoint."""
    return JSONResponse(_global_status)

async def custom_error_handler(request: Request, exc: Exception) -> Response:
    """Custom handler for errors, including 404 HTTPException."""
    # Simplified handler for diagnostics (V2)
    request_path_for_logging = "unknown_path_in_error_handler"
    exc_type_for_logging = type(exc).__name__
    original_exc_str_for_logging = str(exc)

    try:
        request_path_for_logging = request.url.path
    except Exception as e_path:
        logger.error(f"CUSTOM_ERROR_HANDLER_V2: Error getting request.url.path: {e_path}")

    logger.error(
        f"CUSTOM_ERROR_HANDLER_V2: Entered. Path (logging): '{request_path_for_logging}', "
        f"ExcType: {exc_type_for_logging}, ExcStr: '{original_exc_str_for_logging}'"
    )

    status_code_to_set = 500
    response_content = f"Static Server Error processing '{request_path_for_logging}'. Exception: {exc_type_for_logging}."

    if isinstance(exc, HTTPException):
        status_code_to_set = exc.status_code
        if status_code_to_set == 404:
            logger.info(f"Custom error handler (V2): 404 for request path '{request_path_for_logging}'")
            response_content = f"Static 404: Resource at '{request_path_for_logging}' not found."
        else: # Other HTTPExceptions
            exc_detail_str = str(exc.detail) if exc.detail is not None else "No detail"
            response_content = f"Static HTTP Error {status_code_to_set} for '{request_path_for_logging}'. Detail: {exc_detail_str}"

    logger.error(
        f"CUSTOM_ERROR_HANDLER_V2: Determined status_code={status_code_to_set}, "
        f"content='{response_content}'. Creating PlainTextResponse."
    )

    try:
        response_obj = PlainTextResponse(response_content, status_code=status_code_to_set)
        logger.error(f"CUSTOM_ERROR_HANDLER_V2: PlainTextResponse created: {type(response_obj)}. Returning it.")
        return response_obj
    except Exception as e_response:
        logger.critical(f"CUSTOM_ERROR_HANDLER_V2: CRITICAL - FAILED to create PlainTextResponse: {e_response}")
        return PlainTextResponse("Emergency Fallback Error Response", status_code=500)

# Reverted exception_handlers_map to catch all Exceptions with our V2 handler
exception_handlers_map = {
    HTTPException: custom_error_handler,
    404: custom_error_handler,
    500: custom_error_handler,
    Exception: custom_error_handler
}

def create_starlette_app(
    mcp_server: MCPServerSDK[Any],
    allow_origins: list[str] | None = None,
    debug: bool = False,
    stateless: bool = False,
) -> Starlette:
    """Create a Starlette application for the MCP server (single default server mode)."""
    routes, http_manager = create_single_instance_routes(mcp_server, stateless)
    middleware_list: list[Middleware] = []
    if allow_origins:
        middleware_list.append(
            Middleware(
                CORSMiddleware,
                allow_origins=allow_origins,
                allow_methods=["*"],
                allow_headers=["*"],
            ),
        )
    @contextlib.asynccontextmanager
    async def lifespan(_app: Starlette) -> AsyncIterator[None]:
        logger.info("Single-server mode: Lifespan starting...")
        async with http_manager.run():
            logger.info("Single-server mode: StreamableHTTP session manager started.")
            yield
        logger.info("Single-server mode: Lifespan shutting down...")

    return Starlette(
        debug=debug,
        routes=routes,
        middleware=middleware_list,
        lifespan=lifespan,
        exception_handlers=exception_handlers_map
    )

def create_single_instance_routes(
    mcp_server_instance: MCPServerSDK[object],
    stateless_instance: bool,
) -> tuple[list[Route | Mount], StreamableHTTPSessionManager]:
    logger.debug("Creating routes for MCP server instance (stateless: %s)", stateless_instance)
    sse_transport = SseServerTransport("/messages/")
    http_session_manager = StreamableHTTPSessionManager(
        app=mcp_server_instance,
        event_store=None,
        json_response=True,
        stateless=stateless_instance,
    )
    async def handle_sse_instance(request: Request) -> None:
        request_path = request.url.path # Capture for logging in finally/except
        logger.debug(f"SSE HANDLER ({request_path}): ENTERING")
        try:
            logger.debug(f"SSE HANDLER ({request_path}): About to enter sse_transport.connect_sse context.")
            async with sse_transport.connect_sse(
                request.scope, request.receive, request._send,  # noqa: SLF001
            ) as (read_stream, write_stream):
                logger.debug(f"SSE HANDLER ({request_path}): connect_sse context entered. About to run MCP server instance.")
                _update_global_activity()
                await mcp_server_instance.run(
                    read_stream, write_stream, mcp_server_instance.create_initialization_options(),
                )
                logger.debug(f"SSE HANDLER ({request_path}): MCP server instance run COMPLETED.")
            logger.debug(f"SSE HANDLER ({request_path}): Exited sse_transport.connect_sse context normally.")
        except asyncio.CancelledError:
            # This is often expected if the client disconnects.
            logger.warning(f"SSE HANDLER ({request_path}): Task was CANCELLED (likely client disconnect).")
            # Let's try raising a specific HTTPException to see if it's handled better by Starlette from a sub-app.
            logger.warning(f"SSE HANDLER ({request_path}): Raising HTTPException(499) due to client disconnect from CancelledError.")
            raise HTTPException(status_code=499, detail="Client Closed Request (from CancelledError)")
        except TypeError as te:
            logger.error(f"SSE HANDLER ({request_path}): CAUGHT TypeError directly: {te}", exc_info=True)
            # Raise a different HTTPException to distinguish from CancelledError path
            raise HTTPException(status_code=500, detail=f"Internal TypeError during SSE: {te}")
        except Exception as e:
            # Catch other exceptions to see if they originate here.
            logger.error(f"SSE HANDLER ({request_path}): CAUGHT other EXCEPTION during SSE handling: {type(e).__name__}: {e}", exc_info=True)
            # Re-raise to let Starlette's error handling (our custom_error_handler) deal with it.
            raise
        finally:
            logger.debug(f"SSE HANDLER ({request_path}): LEAVING (finally block).")

    async def handle_streamable_http_instance(scope: Scope, receive: Receive, send: Send) -> None:
        logger.debug(f"StreamableHTTP request to {scope.get('path', 'unknown_path')}")
        _update_global_activity()
        await http_session_manager.handle_request(scope, receive, send)
    routes = [
        Mount("/mcp", app=handle_streamable_http_instance),
        Route("/sse", endpoint=handle_sse_instance),
        Mount("/messages/", app=sse_transport.handle_post_message),
    ]
    return routes, http_session_manager

async def run_mcp_server(
    mcp_settings: MCPServerSettings,
    default_server_params: StdioServerParameters | None = None,
    named_server_params: dict[str, StdioServerParameters] | None = None,
) -> None:
    if named_server_params is None:
        named_server_params = {}

    all_routes: list[Route | Mount] = [
        Route("/status", endpoint=_handle_status),
    ]

    async with contextlib.AsyncExitStack() as stack:
        @contextlib.asynccontextmanager
        async def combined_lifespan(_app: Starlette) -> AsyncIterator[None]:
            logger.info("Main application lifespan starting...")
            yield
            logger.info("Main application lifespan shutting down...")

        if default_server_params:
            logger.info(f"Setting up default server: {default_server_params.command} {' '.join(default_server_params.args)}")
            stdio_streams = await stack.enter_async_context(stdio_client(default_server_params))
            session = await stack.enter_async_context(ClientSession(*stdio_streams))
            proxy = await create_proxy_server(session)
            instance_routes, http_manager_default = create_single_instance_routes(proxy, mcp_settings.stateless)
            await stack.enter_async_context(http_manager_default.run())
            all_routes.extend(instance_routes)
            _global_status["server_instances"]["default"] = "configured (fully active)"
            logger.info("Default server available at root paths:")
            logger.info(f"  SSE: http://{mcp_settings.bind_host}:{mcp_settings.port}/sse")
            logger.info(f"  HTTP Messages: http://{mcp_settings.bind_host}:{mcp_settings.port}/messages/")
            logger.info(f"  StreamableHTTP: http://{mcp_settings.bind_host}:{mcp_settings.port}/mcp")

        if named_server_params:
            logger.info(f"Setting up named server mounts.")
            for name, params in named_server_params.items():
                logger.info(f"Configuring named server '{name}': {params.command} {' '.join(params.args)}")

                stdio_streams_named = await stack.enter_async_context(stdio_client(params))
                session_named = await stack.enter_async_context(ClientSession(*stdio_streams_named))
                proxy_named = await create_proxy_server(session_named)
                instance_routes_named, http_manager_named = create_single_instance_routes(proxy_named, mcp_settings.stateless)

                await stack.enter_async_context(http_manager_named.run())
                logger.info(f"StreamableHTTPSessionManager for server '{name}' is active.")

                @contextlib.asynccontextmanager
                async def instance_lifespan(_app: Starlette) -> AsyncIterator[None]:
                    logger.debug(f"Lifespan for sub-app '{name}' starting.")
                    yield
                    logger.debug(f"Lifespan for sub-app '{name}' shutting down.")

                instance_sub_app = Starlette(
                    routes=instance_routes_named,
                    lifespan=instance_lifespan,
                    exception_handlers=exception_handlers_map
                )
                server_mount = Mount(f"/servers/{name}", app=instance_sub_app)
                all_routes.append(server_mount)
                _global_status["server_instances"][name] = "mounted (fully active)"
                logger.info(f"Named server '{name}' mounted at /servers/{name} (fully active)")
                logger.info(f"  SSE: http://{mcp_settings.bind_host}:{mcp_settings.port}/servers/{name}/sse")
                logger.info(f"  HTTP Messages: http://{mcp_settings.bind_host}:{mcp_settings.port}/servers/{name}/messages/")
                logger.info(f"  StreamableHTTP: http://{mcp_settings.bind_host}:{mcp_settings.port}/servers/{name}/mcp")

        if not default_server_params and not named_server_params:
            logger.error("CRITICAL: No servers configured to run. MCP Proxy will not be useful.")
            _global_status["server_instances"] = {"error": "no servers configured to run"}
        elif not named_server_params and default_server_params:
            logger.info("Only default server configured.")
        elif not default_server_params and named_server_params:
            logger.info("Only named servers configured. No default server at root.")

        main_app_middleware: list[Middleware] = []
        if mcp_settings.allow_origins:
            main_app_middleware.append(
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
            middleware=main_app_middleware,
            lifespan=combined_lifespan,
            exception_handlers=exception_handlers_map
        )
        config = uvicorn.Config(
            starlette_app,
            host=mcp_settings.bind_host,
            port=mcp_settings.port,
            log_level=mcp_settings.log_level.lower(),
        )
        http_server = uvicorn.Server(config)
        logger.info(
            "MCP Proxy server starting on %s:%s",
            mcp_settings.bind_host,
            mcp_settings.port,
        )
        await http_server.serve()
