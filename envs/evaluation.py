import pickle
import numpy as np
import torch
from tqdm import tqdm
from envs.pendolf_env import PendolfEnv, PendolfVerifier
from envs.prompts import SYSTEM_PROMPT


def evaluate_agent(model, tokenizer, dataset, device="cuda", batch_size=8, generate_kwargs=None, n_samples=1):
    """Функция для оценки качества обученной модели с использованием PendolfVerifier"""

    # 1. Загрузка датасета через pickle (так как теперь это список Data объектов)
    if isinstance(dataset, str):
        with open(dataset, "rb") as f:
            dataset = pickle.load(f)

    # Инициализация среды и верификатора
    env, verifier = PendolfEnv(), PendolfVerifier()

    # Динамические параметры генерации для pass@k (без изменений)
    if generate_kwargs is None:
        if n_samples > 1:
            generate_kwargs = {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "do_sample": True,
                "num_return_sequences": n_samples,
            }
        else:
            generate_kwargs = {"max_new_tokens": 512, "temperature": 0.0, "do_sample": False}

    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Собираем метрики в списки для усреднения
    metrics = {
        "total": 0,
        "pass_at_k": 0,
        "success": [],
        "total_reward": [],
        "policy_violations": [],
        "invalid_actions": [],
        "tool_calls": [],
        "steps": [],
    }

    for i in tqdm(range(0, len(dataset), batch_size), desc="Evaluating (Batched)"):
        batch_data = dataset[i : i + batch_size]
        metrics["total"] += len(batch_data)

        # 2. Правильное форматирование промпта под чат-шаблон Qwen
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": d.question}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for d in batch_data
        ]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, **generate_kwargs)

        gen_tokens = outputs[:, inputs["input_ids"].shape[1] :]
        responses = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        # 3. Группируем ответы по промптам и скармливаем в Verifier
        for j, data in enumerate(batch_data):
            item_responses = responses[j * n_samples : (j + 1) * n_samples]
            prompt_passed = False

            # Проверяем все сэмплы для конкретного промпта
            for response in item_responses:
                # Вся логика проверок теперь инкапсулирована здесь!
                res = verifier.verify_trajectory(env, data, response.split("\n"))

                metrics["success"].append(res["success"])
                metrics["total_reward"].append(res["total_reward"])
                metrics["policy_violations"].append(res["policy_violations"])
                metrics["invalid_actions"].append(res["invalid_actions"])
                metrics["tool_calls"].append(res["tool_calls"])
                metrics["steps"].append(res["steps"])

                if res["success"]:
                    prompt_passed = True  # Нашли хотя бы один успешный путь!

            if prompt_passed:
                metrics["pass_at_k"] += 1

    total_prompts = metrics["total"] or 1

    result = {
        "success_rate": float(np.mean(metrics["success"])),
        "avg_reward": float(np.mean(metrics["total_reward"])),
        "avg_policy_violations": float(np.mean(metrics["policy_violations"])),
        "avg_invalid_actions": float(np.mean(metrics["invalid_actions"])),
        "avg_tool_calls": float(np.mean(metrics["tool_calls"])),
        "avg_steps": float(np.mean(metrics["steps"])),
        f"pass@{n_samples}": metrics["pass_at_k"] / total_prompts,
    }
    return result
