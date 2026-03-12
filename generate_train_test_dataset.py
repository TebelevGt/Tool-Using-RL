import os, random
from envs import PendolfEnv, PendolfDataset

os.makedirs("data", exist_ok=True)
random.seed(42)
env = PendolfEnv()

# 1. Трейн (солянка из разных уровней)
PendolfDataset.create(env, 500).save_pickle("data/base/pendolf_train.pkl")

# 2. Эвалы (в цикле раскидываем по файлам)
for d, name in zip([2, 6, 9], ["easy", "medium", "hard"]):
    PendolfDataset.create(env, 100, difficulty=d).save_pickle(f"data/base/eval_{name}.pkl")
