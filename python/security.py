import hashlib
import os

# Store a hashed PIN. Default is "49252X" (sha256)
DEFAULT_PIN = "49252X"
PIN_HASH = os.environ.get("ADMIN_PIN_HASH", hashlib.sha256(DEFAULT_PIN.encode()).hexdigest())

def verify_pin(pin: str) -> bool:
    """Check if the provided PIN matches the stored hash."""
    return hashlib.sha256(pin.encode()).hexdigest() == PIN_HASH

def set_pin(new_pin: str):
    """Update the stored PIN hash."""
    global PIN_HASH
    PIN_HASH = hashlib.sha256(new_pin.encode()).hexdigest()