import zmq, json, os
from typing import Mapping, Optional, Tuple, Union
import warnings, pkg_resources
from arkouda import security, io_util
from arkouda.logger import getArkoudaLogger

__all__ = ["verbose", "pdarrayIterThresh", "maxTransferBytes",
           "AllSymbols", "set_defaults", "connect", "disconnect",
           "shutdown", "get_config", "get_mem_used", "__version__",
           "ruok"]

# Try to read the version from the file located at ../VERSION
VERSIONFILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "VERSION")
if os.path.isfile(VERSIONFILE):
    with open(VERSIONFILE, 'r') as f:
        __version__ = f.read().strip()
else:
    # Fall back to the version defined at build time in setup.py
    # pkg_resources is a subpackage of setuptools
    # __package__ is the name of the current package, i.e. "arkouda"
    __version__ = pkg_resources.require(__package__)[0].version

# stuff for zmq connection
pspStr = None
context = zmq.Context()
socket = None
connected = False
# username and token for when basic authentication is enabled
username = None
token = None
# verbose flag for arkouda module
verboseDefVal = False
verbose = verboseDefVal
# threshold for __iter__() to limit comms to arkouda_server
pdarrayIterThreshDefVal = 100
pdarrayIterThresh  = pdarrayIterThreshDefVal
maxTransferBytesDefVal = 2**30
maxTransferBytes = maxTransferBytesDefVal
AllSymbols = "__AllSymbols__"

logger = getArkoudaLogger(name='Arkouda Client')    

# reset settings to default values
def set_defaults() -> None:
    """
    Sets global variables to defaults
    
    Returns
    -------
    None
    """
    global verbose, verboseDefVal, pdarrayIterThresh, pdarrayIterThreshDefVal
    verbose = verboseDefVal
    pdarrayIterThresh  = pdarrayIterThreshDefVal
    maxTransferBytes = maxTransferBytesDefVal

# create context, request end of socket, and connect to it
def connect(server : str="localhost", port : int=5555, timeout : int=0, 
                           access_token : str=None, connect_url=None) -> None:
    """
    Connect to a running arkouda server.

    Parameters
    ----------
    server : str, optional
        The hostname of the server (must be visible to the current
        machine). Defaults to `localhost`.
    port : int, optional
        The port of the server. Defaults to 5555.
    timeout : int, optional
        The timeout in seconds for client send and receive operations.
        Defaults to 0 seconds, whicn is interpreted as no timeout.
    access_token : str, optional
        The token used to connect to an existing socket to enable access to 
        an Arkouda server where authentication is enabled. Defaults to None.
    connect_url : str, optional
        The complete url in the format of tcp://server:port?token=<token_value>
        where the token is optional

    Returns
    -------
    None
    
    Raises
    ------
    ConnectionError 
        Raised if there's an error in connecting to the Arkouda server
    ValueError
        Raised if there's an error in parsing the connect_url parameter
    RuntimeError
        Raised if there is a server-side error

    Notes
    -----
    On success, prints the connected address, as seen by the server. If called
    with an existing connection, the socket will be re-initialized.
    """
    global context, socket, pspStr, connected, verbose, username, token

    logger.debug("ZMQ version: {}".format(zmq.zmq_version()))

    if connect_url:
        url_values = _parse_url(connect_url)
        server = url_values[0]
        port = url_values[1]
        if len(url_values) == 3:
            access_token=url_values[2]
    
    # "protocol://server:port"
    pspStr = "tcp://{}:{}".format(server,port)
    
    # check to see if tunnelled connection is desired. If so, start tunnel
    tunnel_server = os.getenv('ARKOUDA_TUNNEL_SERVER')
    if tunnel_server:
        (pspStr, _) = _start_tunnel(addr=pspStr, tunnel_server=tunnel_server)
    
    logger.debug("psp = {}".format(pspStr))

    # create and configure socket for connections to arkouda server
    socket = context.socket(zmq.REQ) # request end of the zmq connection

    # if timeout is specified, set send and receive timeout params
    if timeout > 0:
        socket.setsockopt(zmq.SNDTIMEO, timeout*1000)
        socket.setsockopt(zmq.RCVTIMEO, timeout*1000)
    
    # set token and username global variables
    username = security.get_username()
    token = _set_access_token(access_token=access_token, connect_string=pspStr)

    # connect to arkouda server
    try:
        socket.connect(pspStr)
    except Exception as e:
        raise ConnectionError(e)

    # send the connect message
    message = "connect"
    logger.debug("[Python] Sending request: {}".format(message))

    # send connect request to server and get the response confirming if
    # the connect request succeeded and, if not not, the error message
    message = _send_string_message(message)
    logger.debug("[Python] Received response: {}".format(message))
    connected = True

    conf = get_config()
    if conf['arkoudaVersion'] != __version__:
        warnings.warn(('Version mismatch between client ({}) and server ({}); ' +
                      'this may cause some commands to fail or behave ' +
                      'incorrectly! Updating arkouda is strongly recommended.').\
                      format(__version__, conf['arkoudaVersion']), RuntimeWarning)

def _parse_url(url : str) -> Tuple[str,int,Optional[str]]:
    """
    Parses the url in the following format if authentication enabled:

    tcp://<hostname/url>:<port>?token=<token>

    If authentication is not enabled, the url is expected to be in the format:

    tcp://<hostname/url>:<port>
    
    Parameters
    ----------
    url : str
        The url string    
    
    Returns
    -------
    Tuple[str,int,Optional[str]]
        A tuple containing the host, port, and token, the latter of which is None 
        if authentication is not enabled for the Arkouda server being accessed
    
    Raises
    ------
    ValueError 
        if the url does not match one of the above formats, if the port is not an 
        integer, or if there's a general string parse error raised in the parsing 
        of the url parameter
    """
    try:
        # split on tcp:// and if missing or malformmed, raise ValueError
        no_protocol_stub = url.split('tcp://')
        if len(no_protocol_stub) < 2:
            raise ValueError(('url must be in form tcp://<hostname/url>:<port>' +
                  ' or tcp://<hostname/url>:<port>?token=<token>'))

        # split on : to separate host from port or port?token=<token>
        host_stub = no_protocol_stub[1].split(':')
        if len(host_stub) < 2:
            raise ValueError(('url must be in form tcp://<hostname/url>:<port>' +
                  ' or tcp://<hostname/url>:<port>?token=<token>'))
        host = host_stub[0]
        port_stub = host_stub[1]
   
        if '?token=' in port_stub:
            port_token_stub = port_stub.split('?token=')
            port = int(port_token_stub[0])
            param_token = port_token_stub[1]
        else:
            port = int(port_stub)
            param_token = None
        return (host, port, param_token)
    except Exception as e:
        raise ValueError(e)

def _set_access_token(access_token : Optional[str], 
                                connect_string : Optional[str]) -> Optional[str]:
    """
    Sets the access_token for the connect request by doing the following:

    1. retrieves the token configured for the connect_string from the 
       .arkouda/tokens.txt file, if any
    2. if access_token is None, returns the retrieved token
    3. if access_token is not None, replaces retrieved token with the access_token
       to account for situations where the token can change for a url (for example,
       the arkouda_server is restarted and a corresponding new token is generated).

    Parameters
    ----------
    username : str
        The username retrieved from the user's home directory    
    access_token : str, optional
        The access_token supplied by the user, which is required if authentication
        is enabled, defaults to None
    connect_string : str, optional
        The arkouda_server host:port connect string, defaults to None. If None, then
        the connect_string is localhost:5555
    
    Returns
    -------
    str
        The access token configured for the host:port
    
    Raises
    ------
    IOError
        If there's an error writing host:port -> access_token mapping to
        the user's tokens.txt file or retrieving the user's tokens
    """
    path = '{}/tokens.txt'.format(security.get_arkouda_client_directory())
    try:
        tokens = io_util.delimited_file_to_dict(path)
    except Exception as e:
        raise IOError(e)

    if access_token and access_token not in {'','None'}:
        saved_token = tokens.get(connect_string)
        if saved_token is None or saved_token != access_token:
            tokens[connect_string] = access_token
            try:
                io_util.dict_to_delimited_file(values=tokens, path=path, 
                                               delimiter=',')
            except Exception as e:
                raise IOError(e)
        return access_token   
    else:
        try:
            tokens = io_util.delimited_file_to_dict(path)
        except Exception as e:
            raise IOError(e)
        return tokens.get(connect_string)

def _start_tunnel(addr : str, tunnel_server : str) -> str:
    """
    Starts ssh tunnel

    Parameters
    ----------
    tunnel_server : str
        The ssh server url   
    
    Returns
    -------
    str
        The new tunneled-version of connect string

    Raises
    ------
    ConnectionError 
        If the ssh tunnel could not be created given the tunnel_server 
        url and credentials (either password or key file)
    """
    from zmq import ssh
    kwargs = {'addr' : addr,
              'server' : tunnel_server}
    keyfile = os.getenv('ARKOUDA_KEY_FILE')
    password = os.getenv('ARKOUDA_PASSWORD')

    if keyfile:
        kwargs['keyfile'] = keyfile
    if password:
        kwargs['password'] = password

    try: 
        return ssh.tunnel.open_tunnel(**kwargs)
    except Exception as e:
        raise ConnectionError(e)

def _send_string_message(message : str, 
                         recv_bytes : bool=False) -> Union[str, bytes]:
    """
    Prepends the message string with Arkouda infrastructure elements 
    including username and authentication token and then sends the 
    resulting, composite string to the Arkouda server.

    Parameters
    ----------
    message : str
        The message including command to be sent to the Arkouda server
    recv_bytes : bool, defaults to False
        A boolean indicating whether the return message will be in bytes
        as opposed to a string

    Returns
    -------
    Union[str,bytes]
        The response string or byte array sent back from the Arkouda server
        
    Raises
    ------
    RuntimeError
        Raised if the return message contains the word "Error", indicating 
        a server-side error was thrown
    """
    message = '{}:{}:{}'.format(username, token, message)

    socket.send_string(message)

    if recv_bytes:
        return_message = socket.recv()
        # raise errors or warnings sent back from the server
        if return_message.startswith(b"Error:"): \
                                   raise RuntimeError(return_message.decode())
        elif return_message.startswith(b"Warning:"): warnings.warn(return_message)
    else:
        return_message = socket.recv_string()
        # raise errors or warnings sent back from the server
        if return_message.startswith("Error:"): raise RuntimeError(return_message)
        elif return_message.startswith("Warning:"): warnings.warn(return_message)
    return return_message

def _send_binary_message(message : bytes, 
                         recv_bytes : bool=False) -> Union[str, bytes]:
    """
    Prepends the binary message with Arkouda infrastructure elements
    including username and authentication token and then sends the
    resulting, composite byte array to the Arkouda server.

    Parameters
    ----------
    message : bytes
        The message including command to be sent to the Arkouda server
    recv_bytes : bool, defaults to False
        A boolean indicating whether the return message will be in bytes
        as opposed to a string

    Returns
    -------
    Union[str,bytes]
        The response string or byte array sent back from the Arkouda server

    Raises
    ------
    RuntimeError
        Raised if the return message contains the word "Error", indicating 
        a server-side error was thrown
    """
    socket.send('{}:{}:'.format(username,token,).encode() + message)

    if recv_bytes:
        return_message = socket.recv()
        # raise errors or warnings sent back from the server
        if return_message.startswith(b"Error:"): \
                                   raise RuntimeError(return_message.decode())
        elif return_message.startswith(b"Warning:"): warnings.warn(return_message)
    else:
        return_message = socket.recv_string()
        # raise errors or warnings sent back from the server
        if return_message.startswith("Error:"): raise RuntimeError(return_message)
        elif return_message.startswith("Warning:"): warnings.warn(return_message)
    return return_message
    
# message arkouda server the client is disconnecting from the server
def disconnect() -> None:
    """
    Disconnects the client from the Arkouda server

    Returns
    -------
    None
    
    Raises
    ------
    ConnectionError
        Raised if there's an error disconnecting from the Arkouda server
    """
    global socket, pspStr, connected, verbose, token

    if socket is not None:
        # send disconnect message to server
        message = "disconnect"
        logger.debug("[Python] Sending request: {}".format(message))
        message = _send_string_message(message)
        logger.debug("[Python] Received response: {}".format(message))
        try:
            socket.disconnect(pspStr)
        except Exception as e:
            raise ConnectionError(e)
        connected = False
    else:
        logger.error("not connected; cannot disconnect")

def shutdown() -> None:
    """
    Sends a shutdown message to the Arkouda server that does the
    following:
    
    1. Delete all objects in the SymTable
    2. Shuts down the Arkouda server
    3. Disconnects the client from the stopped Arkouda Server

    Returns
    -------
    None
    
    Raises
    ------
    RuntimeError
        Raised if the client is not connected to the Arkouda server or
        there is an error in disconnecting from the server
    """
    global socket, pspStr, connected, verbose

    if not connected:
        raise RuntimeError('not connected, cannot shutdown server')
    # send shutdown message to server
    message = "shutdown"

    logger.debug("[Python] Sending request: {}".format(message))
    return_message = _send_string_message(message)
    logger.debug("[Python] Received response: {}".format(return_message))

    try:
        socket.disconnect(pspStr)
    except Exception as e:
        raise RuntimeError(e)
    connected = False

def generic_msg(message : Union[str,bytes], send_bytes : bool=False, 
                recv_bytes : bool=False) -> Union[str, bytes]:
    """
    Sends the binary or string message to the arkouda_server and returns 
    the response sent by the server which is either a success confirmation
    or error message

    Parameters
    ----------
    message : Union[str, bytes]
        The message to be sent in the form of a string or bytes array
    send_bytes : bool
        Indicates if the message to be sent is binary, defaults to False
    recv_bypes : bool
        Indicates if the return message will be binary, default to False

    Returns
    -------
    Union[str, bytes]
        The string or binary return message
    
    Raises
    ------
    KeyboardInterrupt
        Raised if the user interrupts during command execution
    RuntimeError
        Raised if the client is not connected to the server or if
        there is a server-side error thrown
    """
    global socket, pspStr, connected, verbose

    if not connected:
        raise RuntimeError("client is not connected to a server")

    try:
        if send_bytes:
            return _send_binary_message(message=message, 
                                            recv_bytes=recv_bytes)
        else:
            logger.debug("[Python] Sending request: {}".format(message))
            return _send_string_message(message=message, 
                                            recv_bytes=recv_bytes)
    except KeyboardInterrupt as e:
        # if the user interrupts during command execution, the socket gets out 
        # of sync reset the socket before raising the interrupt exception
        socket = context.socket(zmq.REQ)
        socket.connect(pspStr)
        raise e

def get_config() -> Mapping[str, Union[str, int, float]]:
    """
    Get runtime information about the server.

    Returns
    -------
    Mapping[str, Union[str, int, float]]
        serverHostname
        serverPort
        numLocales
        numPUs (number of processor units per locale)
        maxTaskPar (maximum number of tasks per locale)
        physicalMemory
        
    Raises
    ------
    RuntimeError
        Raised if there is a server-side error in getting memory used
    ValueError
        Raised if there's an error in parsing the JSON-formatted server
        configuration into a dict
    """
    json_string = generic_msg("getconfig")

    try:
        return json.loads(json_string)
    except Exception as e:
        raise ValueError(e)

def get_mem_used() -> int:
    """
    Compute the amount of memory used by objects in the server's symbol table.

    Returns
    -------
    int
        Indicates the amount of memory allocated to symbol table objects.
    
    Raises
    ------  
    RuntimeError
        Raised if there is a server-side error in getting memory used
    ValueError
        Raised if the returned value is not an int-formatted string
    """
    mem_used_message = generic_msg("getmemused")
    return int(mem_used_message)

def _no_op() -> str:
    """
    Send a no-op message just to gather round trip time

    Returns
    -------
    str
        The noop command result
        
    Raises
    ------  
    RuntimeError
        Raised if there is a server-side error in executing noop request
    """
    return generic_msg("noop")
  
def ruok() -> str:
    """
    Simply sends an "ruok" message to the server and, if the return message is
    "imok",t his means the arkouda_server is up and operating normally. A return
    message of "imnotok" indicates an error occurred or the connection timed out.
    
    This method is basically a way to do a quick healthcheck in a way that does 
    not require error handling.
    
    Returns
    -------
    str
        A string indicating if the server is operating normally (imok), if there's
        an error server-side, or if ruok did not return a response (imnotok) in
        both of the latter cases
    """
    try:
        res = generic_msg('ruok')
        if res == 'imok':
            return 'imok'
        else:
            return 'imnotok because: {}'.format(res)
    except Exception as e:
        return 'ruok did not return response: {}'.format(str(e))
