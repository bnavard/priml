from priml.baselines.speedrundit.data import SpeedrunDiTData


def test_synthetic_batches_match_contract() -> None:
    data = SpeedrunDiTData.Config(synthetic=True, num_samples=3, batch_size=2, image_size=4, latent_channels=4, cls_token_dim=8).make()
    batch = next(data.train_dataloader())
    assert batch["media"].shape == (2, 4, 4, 4)
    assert batch["label"].shape == (2,)
    assert batch["cls_token"].shape == (2, 8)

