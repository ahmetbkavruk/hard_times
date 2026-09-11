from .config import Config
from .world import World

def main():
    cfg = Config()
    world = World(cfg, seed=7)
    world.run(1000)

    final = world.history[-1]
    print("Endogenous Niche World v0.1")
    print(f"seed        : {world.seed}")
    print(f"steps       : {final['t']}")
    print(f"population  : {final['population']}")
    print(f"lineages    : {final['lineages']}")
    print(
        "mean traits : "
        f"hA={final['mean_hA']:.3f}, "
        f"hB={final['mean_hB']:.3f}, "
        f"hC={final['mean_hC']:.3f}, "
        f"v={final['mean_v']:.3f}"
    )

if __name__ == "__main__":
    main()
