from .spec import SportSpec

# FIVB indoor court: 18 x 9 m, attack lines 3 m from the net.
VOLLEYBALL = SportSpec(
    name="volleyball",
    length=18.0,
    width=9.0,
    free_zone=3.0,
    lines=(
        ((6.0, 0.0), (6.0, 9.0)),
        ((12.0, 0.0), (12.0, 9.0)),
    ),
)
