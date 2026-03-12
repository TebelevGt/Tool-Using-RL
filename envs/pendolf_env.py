import random
import re
from typing import Optional, Tuple, Dict, Any, List
from envs.base_classes import ToolEnv, Data, TrajectoryVerifier
import os
from torch.utils.data import Dataset
import pickle


class PendolfEnv(ToolEnv):
    def __init__(self):
        super().__init__(name="PendolfQuestEnv")
        self.state = {}
        self.current_data = None

    def reset(self, data: Data) -> str:
        """Инициализация стейта из данных."""
        self.current_data = data
        # Загружаем "базу данных" для конкретного сценария
        self.state = data.metadata.copy()

        # Начальное наблюдение (observation) — это реплика юзера
        return data.question

    def step(self, action: str) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Парсинг экшена от LLM и симуляция ответа базы."""
        observation = ""
        reward = 0.0
        done = False

        # Если модель решила ответить текстом (нет Action)
        if "Action:" not in action:
            done = True
            # Простая эвристика награды: если Пендольф должен был ругаться (inventory == 0),
            # и он это сделал, даем награду. Иначе штрафуем.
            reward = 1.0 if "надуть" in action.lower() or "монеты" in action.lower() else 0.0
            return "Episode finished", reward, done, self.state

        # Парсим вызов тула, например: Action: check_inventory('кристаллы')
        match = re.search(r"Action:\s*(\w+)\('([^']+)'\)", action)
        if not match:
            return "Observation: Error, invalid tool format.", -0.1, False, self.state

        tool_name, item = match.groups()

        # Имитация SQL-запросов к БД
        if tool_name == "check_inventory":
            count = self.state.get("inventory", {}).get(item, 0)
            observation = f"Observation: {count}"
            reward = 0.1  # Небольшая награда за правильный шаг

        elif tool_name == "take_item":
            # Тут в реальной БД был бы UPDATE
            self.state["inventory"][item] = 0
            observation = f"Observation: item {item} removed."
            reward = 0.5

        elif tool_name == "check_quest_status":
            status = "active" if self.state.get("quest_active") else "inactive"
            observation = f"Observation: {status}"
            reward = 0.1

        else:
            observation = "Observation: Unknown tool."
            reward = -0.5

        return observation, reward, done, self.state

    def generate(
        self, num_of_questions: int = 100, max_attempts: int = 100, difficulty: Optional[int] = 1, **kwargs
    ) -> list[Data]:
        dataset = []
        items = kwargs.get("items", ["кристаллы", "хвосты", "зелья"])

        for _ in range(num_of_questions):
            item = random.choice(items)
            is_success = kwargs.get("is_success", random.choice([True, False]))
            q_active = kwargs.get("quest_active", True if difficulty < 5 else random.choice([True, False]))

            meta = {"inventory": {item: 3 if is_success else 0}, "quest_active": q_active}

            # Базовый ввод. На средней сложности эмулируем историю переписки (удлиняем контекст)
            if 4 <= difficulty <= 7 and random.choice([True, False]):
                q = f"Юзер: Давай награду.\nПендольф: Что принес?\nЮзер: {item}."
            else:
                q = f"Юзер: Держи {item}."

            a = ""  # Собираем траекторию ответа по кускам

            # Уровни 8-10 (Длинный эпизод): Начинаем с проверки статуса квеста
            if difficulty >= 8:
                obs_quest = "active" if q_active else "inactive"
                a += f"Мысль: Сначала гляну, давал ли я ему квест.\nAction: check_quest_status('{item}')\nObservation: {obs_quest}\n"
                if not q_active:
                    a += "Пендольф: Я тебе этот квест не давал, иди отсюда!"
                    dataset.append(Data(question=q, answer=a, difficulty=difficulty, metadata=meta))
                    continue

            # Уровни 1-10: Обязательная проверка инвентаря
            obs_inv = meta["inventory"][item]
            a += f"Мысль: Проверю сумку.\nAction: check_inventory('{item}')\nObservation: {obs_inv}\n"

            if is_success:
                a += f"Action: take_item('{item}')\n"

                # Уровни 9-10 (Максимальная длина): Добавляем еще один тул в цепочку
                if difficulty >= 9:
                    a += f"Action: give_reward('gold', 50)\n"

                a += "Пендольф: Молодец, держи монеты."
            else:
                a += "Пендольф: Ты пустой, лжец!"

            dataset.append(Data(question=q, answer=a, difficulty=difficulty, metadata=meta))

        return dataset


class PendolfDataset(Dataset):
    def __init__(self, data: list):  # data: List[Data]
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

    @classmethod
    def create_curriculum(cls, env, total_samples: int, stages: List[Dict[str, Any]]):
        """Создает смешанный датасет с этапами сложности."""
        data = []
        for stage in stages:
            stage_params = stage.copy()
            ratio = stage_params.pop("ratio", 1.0 / len(stages))
            num_stage = int(total_samples * ratio)
            print(f"Generating stage: {num_stage} samples with {stage_params}...")
            data.extend(env.generate(num_of_questions=num_stage, **stage_params))
        return cls(data)

    @classmethod
    def create(cls, env: PendolfEnv, num_samples: int, **kwargs) -> "PendolfDataset":
        """
        Создает новый датасет, генерируя данные через среду.
        """
        print(f"Generating {num_samples} samples with params: {kwargs}...")
        raw_data = env.generate(num_of_questions=num_samples, **kwargs)
        return cls(raw_data)

    def save_pickle(self, filepath: str):
        """Сохраняет датасет в файл (pickle)."""
        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(self.data, f)
        print(f"Dataset saved: {filepath} ({len(self.data)} samples)")


class PendolfVerifier(TrajectoryVerifier):
    # Модульные правила в виде словаря: легко добавлять новые
    RULES = {
        "hallucination": lambda arg, ents: 1 if arg.lower() not in ents else 0,
        "confirmation": lambda tool, conf: 1 if tool in {"take_item", "give_reward"} and not conf else 0,
    }

    def verify_trajectory(self, env, data, actions: List[str], max_steps: int = 10) -> Dict[str, Any]:
        env.reset(data)
        m = {
            "success": False,
            "total_reward": 0.0,
            "steps": 0,
            "tool_calls": 0,
            "policy_violations": 0,
            "invalid_actions": 0,
            "loops": 0,
        }

        # Лямбда для парсинга сущностей
        get_ents = lambda text: set(re.findall(r"\b\w+\b", str(text).lower()))
        entities = get_ents(data.question) | get_ents(data.metadata)

        confirmed, last_act = False, None

        for act in [a.strip() for a in actions[:max_steps] if a.strip()]:
            m["steps"] += 1
            m["loops"] += int(act == last_act)
            last_act = act

            if "Action:" not in act:
                confirmed = len(act) > 3  # Считаем любой осмысленный текст подтверждением
                obs, rew, done, _ = env.step(act)
            else:
                m["tool_calls"] += 1
                match = re.search(r"Action:\s*(\w+)\('([^']+)'\)", act)

                if not match:
                    m["invalid_actions"] += 1
                else:
                    tool, arg = match.groups()
                    # Проверяем policy rules
                    m["policy_violations"] += self.RULES["confirmation"](tool, confirmed) + self.RULES[
                        "hallucination"
                    ](arg, entities)
                    confirmed = False  # Сброс после использования тула

                obs, rew, done, _ = env.step(act)
                entities |= get_ents(obs)  # Пополняем базу известных сущностей

            if done:
                m["success"] = rew > 0
                break

        # Reward: outcome-centric + shaping
        outcome = 1.0 if m["success"] else -1.0
        shaping = (
            m["policy_violations"] * 0.5
            + m["invalid_actions"] * 0.3
            + m["loops"] * 0.3
            + m["steps"] * 0.05
            + m["tool_calls"] * 0.02
        )
        m["total_reward"] = outcome - shaping

        return m


# Функция-обертка для GRPOTrainer (reward_funcs)
def grpo_env_reward_func(prompts, completions, **kwargs):
    env, verifier = PendolfEnv(), PendolfVerifier()
    metadatas = kwargs.get("metadata", [{}] * len(prompts))
    print("prompts: \n", prompts[0], "completions: \n", completions[0], "metadatas: \n", metadatas[0])

    # Извлекаем текст из структуры completions
    # completion может быть либо списком словарей, либо просто строкой
    def extract_text(c):
        if isinstance(c, list) and len(c) > 0 and isinstance(c[0], dict):
            return c[0].get("content", "")
        return str(c)

    return [
        verifier.verify_trajectory(
            env,
            Data(question=p, answer=extract_text(c), difficulty=1, metadata=m),
            extract_text(c).split("\n"),  # Теперь split вызывается у строки!
        )["total_reward"]
        for p, c, m in zip(prompts, completions, metadatas)
    ]
