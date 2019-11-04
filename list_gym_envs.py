from gym import envs
all_envs = envs.registry.all()

for a in all_envs:
    print(a)
