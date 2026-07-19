# Data privacy

Accelerometer captures can reveal printer behavior, timestamps, directory names,
host details, and configuration metadata. They are private unless the owner
explicitly chooses to publish them.

Generated raw captures, reports, `.stdata` files, printer configuration, and
credential material must remain outside Git. Public regression tests use only
synthetic signals and non-identifying aggregate metrics. Before sharing an
artifact, inspect both its visible content and embedded metadata.

The software must not send captures or reports over a network. Any future upload
feature requires separate, explicit, informed opt-in and a documented retention
policy.
