from priml.baselines.speedrundit.experiments import exp000, exp001, exp_smoke


def test_experiment_names_and_smoke_geometry() -> None:
    assert exp000().experiment_name == "exp000"
    assert exp001().experiment_name == "exp001"
    smoke = exp_smoke()
    assert smoke.dataset.synthetic
    assert smoke.step.model.channels_hidden == 16

