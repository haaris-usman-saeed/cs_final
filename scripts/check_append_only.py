"""Fail unless NEW log extends OLD log byte-for-byte and its hash chain verifies. Usage: check_append_only.py OLD NEW"""
import sys
from pathlib import Path

from fireaq.prospective import verify_chain

old, new = Path(sys.argv[1]), Path(sys.argv[2])
o = old.read_bytes() if old.exists() else b''
n = new.read_bytes() if new.exists() else b''
if not n.startswith(o):
    sys.exit('APPEND-ONLY VIOLATION: existing prospective records were modified or removed')
chk = verify_chain(new)
if not chk['ok']:
    sys.exit(f'HASH CHAIN BROKEN at line {chk["broken_at_line"]}')
print(f'append-only OK: {len(o.splitlines())} -> {len(n.splitlines())} records')
