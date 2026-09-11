from ht import Config, World

cfg = Config(
    L=40,
    initial_agents=250,
)

world = World(cfg, seed=7)
world.run(1000)

for row in world.history[::100]:
    print(
        f"t={row['t']:4d}  "
        f"N={row['population']:4d}  "
        f"lineages={row['lineages']:4d}  "
        f"births={row['births']:5d}  "
        f"gen={row['mean_generation']:.1f}/{row['max_generation']:2d}  "
        f"<hA>={row['mean_hA']:.3f}  "
        f"<s>={row['mean_production_allocation']:.3f}  "
        f"<v>={row['mean_v']:.3f}"
    )

print("final:", world.history[-1])
