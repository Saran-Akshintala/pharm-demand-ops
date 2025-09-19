"""
Streamlit compatibility utilities to handle different versions gracefully.
"""

import streamlit as st
import inspect

def safe_download_button(*args, **kwargs):
    """
    Safe wrapper for st.download_button that handles version compatibility.
    Removes unsupported parameters for older Streamlit versions.
    """
    # Get the signature of st.download_button
    sig = inspect.signature(st.download_button)
    supported_params = set(sig.parameters.keys())
    
    # Filter out unsupported parameters
    safe_kwargs = {k: v for k, v in kwargs.items() if k in supported_params}
    
    # Call st.download_button with only supported parameters
    return st.download_button(*args, **safe_kwargs)

def safe_button(*args, **kwargs):
    """
    Safe wrapper for st.button that handles version compatibility.
    Removes unsupported parameters for older Streamlit versions.
    """
    # Get the signature of st.button
    sig = inspect.signature(st.button)
    supported_params = set(sig.parameters.keys())
    
    # Filter out unsupported parameters
    safe_kwargs = {k: v for k, v in kwargs.items() if k in supported_params}
    
    # Call st.button with only supported parameters
    return st.button(*args, **safe_kwargs)

def get_streamlit_version():
    """Get the current Streamlit version."""
    try:
        return st.__version__
    except AttributeError:
        return "unknown"

def is_streamlit_version_at_least(min_version):
    """Check if current Streamlit version is at least the specified version."""
    try:
        from packaging import version
        current_version = get_streamlit_version()
        if current_version == "unknown":
            return False
        return version.parse(current_version) >= version.parse(min_version)
    except ImportError:
        # If packaging is not available, assume older version
        return False
