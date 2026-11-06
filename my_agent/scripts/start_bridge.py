import sys
sys.path.insert(0, '/home/udhayan/.local/share/FreeCAD/v1-1/Mod/RobustMCPBridge')

try:
    from freecad_mcp_bridge.server import FreecadMCPPlugin
    from preferences import get_socket_port, get_xmlrpc_port
    from freecad_mcp_bridge.bridge_utils import register_mcp_plugin

    xmlrpc_port = get_xmlrpc_port()
    socket_port = get_socket_port()
    plugin = FreecadMCPPlugin(
        host="localhost",
        port=socket_port,
        xmlrpc_port=xmlrpc_port,
        enable_xmlrpc=True,
    )
    plugin.start()
    register_mcp_plugin(plugin, xmlrpc_port, socket_port)
    print(f"Bridge started successfully on port {xmlrpc_port}")
except Exception as e:
    print(f"Error starting bridge: {e}")
