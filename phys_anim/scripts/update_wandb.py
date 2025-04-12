import wandb
import multiprocessing

# Set your project name here
PROJECT_NAME = "task_tokens/eval_results_reach"
USE_MULTIPROCESSING = True  # Set to False to run sequentially


def update_run(run):
    """Updates a single W&B run with the new 'algo_clean_name' column."""
    algo_type = run.config.get("algo_type", "")
    clean_exp_name = run.config.get("clean_exp_name", "")

    if not algo_type:
        algo_type = "PureRL" if run.config.get("disable_discriminator") else "AMP"

    new_column_value = f"{algo_type}_{clean_exp_name}"
    run.summary["algo_clean_name"] = new_column_value
    run.update()

    print(f"Updated run {run.id} with algo_clean_name: {new_column_value}")


def main():
    """Fetches W&B runs and updates them, optionally using multiprocessing."""
    api = wandb.Api()
    runs = api.runs(PROJECT_NAME)

    if USE_MULTIPROCESSING:
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            pool.map(update_run, runs)
    else:
        for run in runs:
            update_run(run)


if __name__ == "__main__":
    main()
