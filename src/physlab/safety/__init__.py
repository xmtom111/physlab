"""Real-world policy and the approval gate.

``gate`` rejects a real-world action that has no approval token. Checking
that the token belongs to a specific plan or protocol is T12. Agents must
not edit safety configs or approve their own real-world actions.
"""

from physlab.safety.gate import gate

__all__ = ["gate"]
