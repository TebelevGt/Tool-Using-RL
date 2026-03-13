import datasets
import pickle
from envs.prompts import SYSTEM_PROMPT


def get_pendolf_dataset(filepath: str):
    """Загружает PendolfDataset и готовит его для GRPOTrainer."""

    # Загружаем твои сохраненные данные (Data объекты)
    with open(filepath, "rb") as f:
        raw_data = pickle.load(f)

    data_dict = {
        "prompt": [],  # Список сообщений (system + user)
        "metadata": [],  # Словари с состоянием мира [cite: 26, 94]
        "difficulty": [],  # Уровень сложности [cite: 18, 72]
    }

    for item in raw_data:
        # 1. Формируем промпт в формате чата
        chat_prompt = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": item.question}]
        data_dict["prompt"].append(chat_prompt)

        # 2. Прокидываем метаданные для верификатора [cite: 82, 94]
        data_dict["metadata"].append(item.metadata)

        # 3. Сохраняем сложность (целое число 1-10) [cite: 18, 76]
        data_dict["difficulty"].append(item.difficulty)

    return datasets.Dataset.from_dict(data_dict)
