"""``python -m physlab.mcp_server`` serves stdio using the default config."""

from physlab.mcp_server.server import serve

if __name__ == "__main__":
    serve("stdio")
