from src.data import load_foods
from src.simulated_annealing import simulated_annealing
from src.initialize import greedy_init
import argparse
from src.utils import print_plan


def _parse_args():
    parser = argparse.ArgumentParser(description="Simulated Annealing Meal Planner")

    # Data
    parser.add_argument(
        "--foods_path",
        type=str,
        default="data/processed/foods.csv",
        help="path to processed foods CSV",
    )

    # Logged intake
    parser.add_argument(
        "--energy", type=float, default=400.0, help="logged energy (kcal)"
    )
    parser.add_argument(
        "--protein", type=float, default=15.0, help="logged protein (g)"
    )
    parser.add_argument("--fat", type=float, default=12.0, help="logged fat (g)")
    parser.add_argument("--carbs", type=float, default=55.0, help="logged carbs (g)")
    parser.add_argument("--fiber", type=float, default=5.0, help="logged fiber (g)")
    parser.add_argument(
        "--sodium", type=float, default=500.0, help="logged sodium (mg)"
    )
    parser.add_argument("--sugar", type=float, default=10.0, help="logged sugar (g)")

    # Initial State
    parser.add_argument(
        "--greedy_init",
        action="store_true",
        help="generate initial state automatically using greedy initialization (recommended)",
    )
    parser.add_argument(
        "--initial_state",
        type=str,
        nargs="*",
        default=None,
        help="manual initial state as food_id,multiplier pairs e.g. --initial_state 42,1.5 137,1.0 88,0.5 "
             "(ignored if --greedy_init is set)",
    )
    parser.add_argument(
        "--candidate_pool_size",
        type=int,
        default=50,
        help="number of candidate foods evaluated per slot during greedy init (higher = better quality, slower)",
    )

    # Daily targets
    parser.add_argument(
        "--target_energy", type=float, default=2000.0, help="daily energy target (kcal)"
    )
    parser.add_argument(
        "--target_protein", type=float, default=50.0, help="daily protein target (g)"
    )
    parser.add_argument(
        "--target_fat", type=float, default=65.0, help="daily fat target (g)"
    )
    parser.add_argument(
        "--target_carbs", type=float, default=260.0, help="daily carbs target (g)"
    )
    parser.add_argument(
        "--target_fiber", type=float, default=28.0, help="daily fiber target (g)"
    )

    # Upper limits
    parser.add_argument(
        "--limit_sodium",
        type=float,
        default=2300.0,
        help="daily sodium upper limit (mg)",
    )
    parser.add_argument(
        "--limit_sugar", type=float, default=50.0, help="daily sugar upper limit (g)"
    )

    # Penalty weights
    parser.add_argument(
        "--w_gap", type=float, default=1.0, help="weight for nutrition gap penalty"
    )
    parser.add_argument(
        "--w_exceed", type=float, default=5.0, help="weight for exceeding upper limits"
    )
    parser.add_argument(
        "--w_variety",
        type=float,
        default=2.0,
        help="weight for category repetition penalty",
    )
    parser.add_argument(
        "--w_portion", type=float, default=0.5, help="weight for portion size penalty"
    )
    parser.add_argument(
        "--w_balance", type=float, default=1.0, help="weight for calorie balance penalty"
    )

    # Portion bounds
    parser.add_argument(
        "--portion_min",
        type=float,
        default=0.5,
        help="minimum serving multiplier (0.5 = 50g)",
    )
    parser.add_argument(
        "--portion_max",
        type=float,
        default=5.0,
        help="maximum serving multiplier (5.0 = 500g)",
    )

    # Slots
    parser.add_argument(
        "--slots",
        type=str,
        nargs="*",
        default=["breakfast", "lunch", "dinner", "snack"],
        help="meal slots to plan",
    )
    parser.add_argument(
        "--slots_done",
        type=str,
        nargs="*",
        default=["breakfast"],
        help="slots already eaten e.g. --slots_done breakfast lunch",
    )

    # SA hyperparameters
    parser.add_argument(
        "--t_start", type=float, default=500.0, help="SA starting temperature"
    )
    parser.add_argument(
        "--t_end", type=float, default=0.5, help="SA ending temperature"
    )
    parser.add_argument("--alpha", type=float, default=0.995, help="SA cooling rate")
    parser.add_argument(
        "--max_steps", type=int, default=10000, help="SA number of steps"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    return parser.parse_args()


def main():
    args = _parse_args()

    logged_today = {
        "energy_kcal": args.energy,
        "protein_g": args.protein,
        "fat_g": args.fat,
        "carbs_g": args.carbs,
        "fiber_g": args.fiber,
        "sodium_mg": args.sodium,
        "sugar_g": args.sugar,
    }

    targets = {
        "energy_kcal": args.target_energy,
        "protein_g": args.target_protein,
        "fat_g": args.target_fat,
        "carbs_g": args.target_carbs,
        "fiber_g": args.target_fiber,
    }

    upper_limits = {
        "sodium_mg": args.limit_sodium,
        "sugar_g": args.limit_sugar,
    }

    sa_cfg = {
        "T_start": args.t_start,
        "T_end": args.t_end,
        "alpha": args.alpha,
        "max_steps": args.max_steps,
        "seed": args.seed,
    }

    penalty_cfg = {
        "w_gap": args.w_gap,
        "w_exceed": args.w_exceed,
        "w_variety": args.w_variety,
        "w_portion": args.w_portion,
        "w_balance": args.w_balance,
        "portion_min": args.portion_min,
        "portion_max": args.portion_max,
    }

    foods = load_foods(args.foods_path)

    remaining_slots = [s for s in args.slots if s not in args.slots_done]

    if args.greedy_init:
        initial_state = greedy_init(
            foods_df=foods,
            logged=logged_today,
            targets=targets,
            n_slots=len(remaining_slots),
            penalty_cfg=penalty_cfg,
            candidate_pool_size=args.candidate_pool_size,
            seed=args.seed,
        )
    else:
        if args.initial_state is None:
            parser_err = (
                "No initial state provided. Pass --greedy_init to generate one "
                "automatically, or supply --initial_state as food_id,multiplier pairs."
            )
            raise ValueError(parser_err)
        initial_state = [
            (int(s.split(",")[0]), float(s.split(",")[1])) for s in args.initial_state
        ]
        if len(initial_state) != len(remaining_slots):
            raise ValueError(
                f"--initial_state has {len(initial_state)} entries but there are "
                f"{len(remaining_slots)} remaining slots {remaining_slots}. "
                f"These must match, or use --greedy_init instead."
            )

    best_state = simulated_annealing(
        foods, logged_today, initial_state, targets, upper_limits, sa_cfg, penalty_cfg
    )
    print_plan(best_state, remaining_slots, foods, targets, upper_limits, logged_today)


if __name__ == "__main__":
    main()