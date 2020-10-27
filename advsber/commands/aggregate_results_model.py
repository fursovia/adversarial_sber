from pathlib import Path
import json

import typer
import pandas as pd


def main(results_path: Path, filename: str = "output_models.json"):
    paths = results_path.rglob(filename)
    metrics = []
    for path in paths:
        with open(str(path)) as f:
            curr_metrics = json.load(f)
            curr_metrics["model_name"] = path.parent.name
            curr_metrics["model_type"] = path.parent.parent.name
            curr_metrics["dataset"] = path.parent.parent.parent.parent.name
            metrics.append(curr_metrics)

    metrics = pd.DataFrame(metrics)
    typer.echo(metrics.to_markdown())
    output_path = str(results_path / "metrics_models.csv")
    typer.secho(f"Saving results to {output_path}", fg="green")
    metrics.to_csv(output_path, index=False)


if __name__ == "__main__":
    typer.run(main)
