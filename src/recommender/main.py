from recommender.adapters import VanillaAdapter

if __name__ == "__main__":
    adapter = VanillaAdapter()
    train_config = {
        "model_name_or_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "training_data_path": "ought/raft",
        "tuning_strategy": "full",
    }
    ir_to_apply, json_patches = adapter.execute(
        train_config,
        compute_config={},
        dist_config={},
        data_config={},
        unique_tag="gpq12df",
    )
    print(ir_to_apply, json_patches)
