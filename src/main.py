from tvl import tvl
from tvl import tvl_struct


def main():
    struct = tvl_struct()

    sim = tvl(struct)

    sim.run()


main()
