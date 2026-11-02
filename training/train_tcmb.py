"""Create an auditable TCMB heuristic configuration.

This initializes the current rule-based prototype. It does not claim to train a
neural budget predictor; learned training requires validated supervision.
"""
import argparse, json
from pathlib import Path
from tcmb.typological_features import PROFILES
from tcmb.budget_predictor import TypologyConditionedBudgetPredictor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/tcmb_config.json")
    parser.add_argument("--base-budget", type=int, default=5)
    parser.add_argument("--max-budget", type=int, default=20)
    args = parser.parse_args()
    predictor = TypologyConditionedBudgetPredictor(
        base_budget=args.base_budget, max_budget=args.max_budget
    )
    config = {
        "method": "heuristic_typology_conditioned_budget_prototype",
        "base_budget": args.base_budget,
        "max_budget": args.max_budget,
        "languages": {code: profile.__dict__ for code, profile in PROFILES.items()},
        "warning": "Heuristic initialization only; not a learned model or validated tokenizer retrofit."
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved prototype configuration to {out}")
    print("Example Tamil budget:", predictor.predict("tam_Taml", "இது ஒரு சோதனை வாக்கியம்."))

if __name__ == "__main__":
    main()
