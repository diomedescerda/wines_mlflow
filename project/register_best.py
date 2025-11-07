import mlflow

# Encontrar el mejor run (máximo r2_score)
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("wines_rf_tuning")
best_run = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.r2_score DESC"],
    max_results=1
)[0]

if best_run:
    print(f"Mejor run: {best_run.info.run_id} con R² = {best_run.data.metrics['r2_score']:.4f}")

    model_uri = f"runs:/{best_run.info.run_id}/model"
    model_details = mlflow.register_model(
        model_uri = model_uri,
        name = "WineQuality"
    )
    print(f"Modelo registrado como versión {model_details.version} en 'WineQuality'")
    client.set_registered_model_alias("WineQuality", "champion", 1)

else:
    print("No se encontraron runs.")