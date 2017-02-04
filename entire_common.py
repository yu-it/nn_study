import os
def isClient():
    return os.environ.get("COMPUTERNAME") == "VAIO"