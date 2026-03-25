import copy
import random
from dataclasses import dataclass
import pandas as pd

from .score import energy
from .initialize import greedy_init


State = list[tuple[int, float]]


@dataclass
class Individual:
    state: State
    score: float


def _random_state(
    foods_df: pd.DataFrame,
    n_slots: int,
    portion_min: float,
    portion_max: float,
    rng: random.Random,
) -> State:
    state: State = []
    for _ in range(n_slots):
        food_id = rng.randrange(len(foods_df))
        mult = round(rng.uniform(portion_min, portion_max), 2)
        state.append((food_id, mult))
    return state


def _mutate(
    state: State,
    foods_df: pd.DataFrame,
    score_cfg: dict[str, float],
    rng: random.Random,
) -> State:
    s = copy.deepcopy(state)
    idx = rng.randrange(len(s))
    move = rng.choice(["replace", "resize", "replace_and_resize"])

    food_id, mult = s[idx]

    if move in ["replace", "replace_and_resize"]:
        food_id = rng.randrange(len(foods_df))

    if move in ["resize", "replace_and_resize"]:
        delta = rng.uniform(-0.5, 0.5)
        mult = round(
            max(
                score_cfg["portion_min"],
                min(score_cfg["portion_max"], mult + delta),
            ),
            2,
        )

    s[idx] = (food_id, mult)
    return s


def _crossover(
    parent1: State,
    parent2: State,
    rng: random.Random,
) -> tuple[State, State]:
    child1: State = []
    child2: State = []

    for gene1, gene2 in zip(parent1, parent2):
        if rng.random() < 0.5:
            child1.append(gene1)
            child2.append(gene2)
        else:
            child1.append(gene2)
            child2.append(gene1)

    return child1, child2


def _tournament_select(
    population: list[Individual],
    tournament_size: int,
    rng: random.Random,
) -> Individual:
    contenders = rng.sample(population, min(tournament_size, len(population)))
    return min(contenders, key=lambda x: x.score)


def genetic_algorithm(
    foods_df: pd.DataFrame,
    logged: dict[str, float],
    targets: dict[str, float],
    upper_limits: dict[str, float],
    remaining_slots: list[str],
    score_cfg: dict[str, float],
    n_slots: int,
    ga_cfg: dict[str, float | int],
    preferences: dict[str, set[str]] | None = None,
    logged_categories: list[str] | None = None,
    follow_history: dict[str, dict[str, int]] | None = None,
) -> State:
    rng = random.Random(int(ga_cfg["seed"]))

    pop_size = int(ga_cfg["pop_size"])
    generations = int(ga_cfg["generations"])
    crossover_rate = float(ga_cfg["crossover_rate"])
    mutation_rate = float(ga_cfg["mutation_rate"])
    elite_size = int(ga_cfg["elite_size"])
    tournament_size = int(ga_cfg["tournament_size"])
    greedy_fraction = float(ga_cfg["greedy_fraction"])
    candidate_pool_size = int(ga_cfg["candidate_pool_size"])

    population: list[Individual] = []

    n_greedy = int(pop_size * greedy_fraction)
    n_random = pop_size - n_greedy

    for i in range(n_greedy):
        state = greedy_init(
            foods_df=foods_df,
            logged=logged,
            targets=targets,
            n_slots=n_slots,
            penalty_cfg=score_cfg,
            candidate_pool_size=candidate_pool_size,
            seed=int(ga_cfg["seed"]) + i,
        )
        score = energy(
            state,
            remaining_slots,
            foods_df,
            logged,
            targets,
            upper_limits,
            score_cfg,
            preferences=preferences,
            logged_categories=logged_categories,
            follow_history=follow_history,
        )
        population.append(Individual(state=state, score=score))

    for _ in range(n_random):
        state = _random_state(
            foods_df,
            n_slots,
            score_cfg["portion_min"],
            score_cfg["portion_max"],
            rng,
        )
        score = energy(
            state,
            remaining_slots,
            foods_df,
            logged,
            targets,
            upper_limits,
            score_cfg,
            preferences=preferences,
            logged_categories=logged_categories,
            follow_history=follow_history,
        )
        population.append(Individual(state=state, score=score))

    best = min(population, key=lambda x: x.score)
    best_state: State = copy.deepcopy(best.state)
    best_score = best.score

    for gen in range(generations):
        population.sort(key=lambda x: x.score)
        next_population: list[Individual] = []

        for i in range(min(elite_size, len(population))):
            next_population.append(
                Individual(
                    state=copy.deepcopy(population[i].state),
                    score=population[i].score,
                )
            )

        while len(next_population) < pop_size:
            parent1 = _tournament_select(population, tournament_size, rng).state
            parent2 = _tournament_select(population, tournament_size, rng).state

            if rng.random() < crossover_rate:
                child1, child2 = _crossover(parent1, parent2, rng)
            else:
                child1 = copy.deepcopy(parent1)
                child2 = copy.deepcopy(parent2)

            if rng.random() < mutation_rate:
                child1 = _mutate(child1, foods_df, score_cfg, rng)
            if rng.random() < mutation_rate:
                child2 = _mutate(child2, foods_df, score_cfg, rng)

            score1 = energy(child1, remaining_slots, foods_df, logged, targets, upper_limits, score_cfg, preferences=preferences, logged_categories=logged_categories, follow_history=follow_history)
            next_population.append(Individual(state=child1, score=score1))

            if len(next_population) < pop_size:
                score2 = energy(child2, remaining_slots, foods_df, logged, targets, upper_limits, score_cfg, preferences=preferences, logged_categories=logged_categories, follow_history=follow_history)
                next_population.append(Individual(state=child2, score=score2))

        population = next_population

        gen_best = min(population, key=lambda x: x.score)
        if gen_best.score < best_score:
            best_state = copy.deepcopy(gen_best.state)
            best_score = gen_best.score

        if gen % 10 == 0:
            print(f"generation {gen}, best_energy={best_score:.4f}")

    return best_state