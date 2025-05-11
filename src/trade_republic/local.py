from server import mcp

import features.lookups
import features.query_data
import features.test

if __name__ == "__main__":
    mcp.run(transport="sse")