# House-in-Order census instrument — FROZEN ARTEFACTS

**Not production code. Not a gate. Not programme authority.**

These are the exact bytes of the document-census instrument developed
under House-in-Order H0 and proposed for freeze at D333/D334. They are
banked here so a future thread can recover the *exact instrument*, not
merely a hash prefix recorded in prose.

Static analysis here is **corroborative and lower-bound only**. It may
never assign GENERATION, authority, or a negative claim beyond the
constructive-exclusion rules it implements.

`MANIFEST.sha256` carries the full SHA-256 of every artefact and a
deterministic aggregate over the set. Verify with:

    cd kai-pm/house_in_order_instrument
    sha256sum -c MANIFEST.sha256

Declared blind spots are recorded in DECISIONS.md at D333/D334 and are
part of the freeze, not omissions from it.
