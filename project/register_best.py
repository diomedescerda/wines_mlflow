import mlflow

# Encontrar el mejor run (máximo r2_score)
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("wines_rf_tuning")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.r2_score DESC"],
    max_results=2
)

if runs:
    print(f"Mejor run: {runs[0].info.run_id} con R² = {runs[0].data.metrics['r2_score']:.4f}")

    model_uri = f"runs:/{runs[0].info.run_id}/model"
    model_details = mlflow.register_model(
        model_uri = model_uri,
        name = "WineQuality"
    )
    print(f"Modelo registrado como versión {model_details.version} en 'WineQuality'")
    client.set_registered_model_alias("WineQuality", "champion", 1)

    # Registering second best model
    model_uri = f"runs:/{runs[1].info.run_id}/model"
    model_details = mlflow.register_model(
        model_uri = model_uri,
        name = "WineQuality"
    )
    print(f"Modelo registrado como versión {model_details.version} en 'WineQuality'")
    client.set_registered_model_alias("WineQuality", "challenger", 2)


else:
    print("No se encontraron runs.")

client.transition_model_version_stage(
    name="WineQuality",
    version=1,
    stage="Production"
)

client.transition_model_version_stage(
    name="WineQuality",
    version=2,
    stage="Staging"
)