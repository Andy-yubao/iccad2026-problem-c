"""Frozen checkpoint identity."""
import hashlib


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
