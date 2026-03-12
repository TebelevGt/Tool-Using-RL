from envs import PendolfEnv, PendolfDataset


def main():
    env = PendolfEnv()

    stages = [
        # 1. Разминка: 20% простых задач (1 тул: сразу проверяем сумку)
        {"difficulty": 2, "ratio": 0.2},
        # 2. Основная часть: 50% средних задач (Контекст с историей переписки)
        {"difficulty": 6, "ratio": 0.5},
        # 3. Челлендж: 30% сложных задач (Длинная цепочка: check_quest -> check_inv -> reward)
        {"difficulty": 9, "ratio": 0.3},
    ]

    total_samples = 500
    output_path = "data/curriculum/pendolf_train_curriculum.pkl"

    print(f"Генерируем Curriculum датасет для Пендольфа ({total_samples} примеров)...")
    dataset = PendolfDataset.create_curriculum(env, total_samples, stages)
    dataset.save_pickle(output_path)
    print(f"Готово! Датасет сохранен в {output_path}")


if __name__ == "__main__":
    main()
