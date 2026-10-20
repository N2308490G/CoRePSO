"""Small CPU usage example; its output is not a manuscript experiment."""
import argparse
import json

import torch
from corepso import minimize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-fes", type=int, default=8000)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    center = [0.4, -0.7, 1.1, 0.2, -1.3, 0.9]

    def objective(x):
        target = x.new_tensor(center)
        weights = x.new_tensor([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
        return ((x - target).square() * weights).sum(dim=-1)

    result = minimize(objective, [-3.0] * 6, [4.0] * 6,
                      max_fes=args.max_fes, seed=42, lineages=4,
                      particles=8, device=args.device)
    print(json.dumps({"x": result.x.tolist(), "fun": result.fun,
                      "nfev": result.nfev}, indent=2))


if __name__ == "__main__":
    main()
