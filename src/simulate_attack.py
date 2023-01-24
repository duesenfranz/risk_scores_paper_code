"""
Simulate an attack on models set in ``experiment_settings.py`` and write the
results to ``data/minimal_distances.csv``.
"""


from collections import defaultdict
from time import time
import foolbox as fb
from robustbench.utils import load_model
from robustbench.data import load_cifar10
import numpy as np
import pandas as pd
import experiment_settings


def main():
    x_test, y_test = load_cifar10(n_examples=experiment_settings.SAMPLES_PER_MODEL)
    attacks = [
        fb.attacks.LinfPGD(),
        fb.attacks.LinfAdamPGD(),
        fb.attacks.LinfDeepFoolAttack(),
    ]
    models = {
        model_name: load_model(model_name, dataset="cifar10", threat_model="Linf")
        for model_name in experiment_settings.MODEL_NAMES
    }
    model_minimal_distances = {}
    attack_name_to_worst_case_counter = defaultdict(lambda: 0)
    start_time = time()
    for model_name, model in models.items():
        model_start_time = time()
        print(f"Analyzing '{model_name}'...")
        fmodel = fb.PyTorchModel(model, bounds=(0, 1))
        epsilons = [0.0, 0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
        advs_all = None
        success_all = None
        minimal_distances = []
        attack_name = []
        for attack in attacks:
            attack_start_time = time()
            print(f"Running attack '{attack}' for {len(epsilons)} different epsilons.")
            _, advs, success = attack(fmodel, x_test, y_test, epsilons=epsilons)
            if advs_all is None:
                advs_all = np.array([a.numpy() for a in advs])
                success_all = success
            else:
                advs_all = np.concatenate(
                    [advs_all, np.array([a.numpy() for a in advs])]
                )
                success_all = np.concatenate([success_all, success])
            attack_name = attack_name + [str(attack) for _ in range(len(advs_all))]
            print(f"Attack took {time() - attack_start_time} seconds.")
        for adversarial_examples, original_observation, are_successfull in zip(
            advs_all.transpose([1, 0, 2, 3, 4]),
            x_test,
            success_all.T,
        ):
            minimal_distance = np.infty
            most_successful = None
            for adversarial_example, is_successful, attack in zip(
                adversarial_examples, are_successfull, attack_name
            ):
                if is_successful:
                    current_distance = np.linalg.norm(
                        (adversarial_example - original_observation.numpy()).flatten(),
                        np.infty,
                    )
                    if current_distance < minimal_distance:
                        most_successful = attack
                        minimal_distance = current_distance
            if most_successful:
                attack_name_to_worst_case_counter[most_successful] = (
                    attack_name_to_worst_case_counter[most_successful] + 1
                )
                minimal_distances.append(minimal_distance)
        model_minimal_distances[model_name] = minimal_distances
        print(
            f"Analyzing model '{model_name}' took {time() - model_start_time} seconds."
        )
    print(model_minimal_distances)
    print(attack_name_to_worst_case_counter)
    print(f"Total analysis took {time() - start_time} seconds")
    df = (
        pd.concat(
            {
                model: pd.DataFrame(dict(distance=dists))
                for model, dists in model_minimal_distances.items()
            }
        )
        .reset_index(level=0)
        .rename({"level_0": "model"}, axis=1)
    )
    df.to_csv(experiment_settings.OUT_PATH, index=False)
    print(f"Wrote data to '{experiment_settings.OUT_PATH}'")


if __name__ == "__main__":
    main()
