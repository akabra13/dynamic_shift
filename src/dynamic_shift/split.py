"""Development / lockbox split for the v2 tests.

A quarter of competition-seasons are fixed by a hash of their key and are
never touched while developing or choosing anything. They are scored once,
at the end, and only if a test passes on the development seasons.
"""
import hashlib


def is_lockbox(key: str) -> bool:
    return int(hashlib.md5(str(key).encode()).hexdigest(), 16) % 4 == 0
