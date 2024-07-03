import sys

major, minor, _, _, _ = sys.version_info
major_req = 3
minor_req = 11
error_message = f'Error! Must be on Python {major_req}.{minor_req} or later.'
if major < major_req or (major == major_req and minor < minor_req):
    sys.exit(error_message)
else:
    sys.exit(0)
