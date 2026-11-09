"""SkyGuard passive sensor bridge for PixEagle.

Detection-only adapter: turns PixEagle-style detections/tracks into
``uas_visual_track`` events for the SkyGuard Evidence API.

This package never commands a vehicle, gimbal or PTZ head. It does not import
MAVSDK, MAVLink, the PixEagle follower or the offboard commander. See
``safety.py`` and README_SKYGUARD.md.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
