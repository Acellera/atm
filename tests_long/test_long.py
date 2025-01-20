import shutil
import os
import pytest


curr_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize("jobname", ["QB_A08_A07", "QB_A15_A14", "QB_A16_A06"])
def _test_regression(tmp_path, jobname):
    from atm.rbfe_structprep import rbfe_structprep
    from atm.rbfe_production import rbfe_production
    from atm.uwham import calculate_uwham

    exp_ddG = {
        "QB_A08_A07": 0.78,
        "QB_A15_A14": 1.28,
        "QB_A16_A06": -1.23,
    }[jobname]
    expected_error = {
        "QB_A08_A07": 0.50,
        "QB_A15_A14": 0.35,
        "QB_A16_A06": 0.35,
    }[jobname]

    shutil.copytree(os.path.join(curr_dir, jobname), os.path.join(tmp_path, jobname))
    rbfe_structprep(os.path.join(tmp_path, jobname, f"{jobname}_asyncre.yaml"))
    rbfe_production(os.path.join(tmp_path, jobname, f"{jobname}_asyncre.yaml"))

    ddG = calculate_uwham(os.path.join(tmp_path, jobname), jobname, 100)[0]

    assert (
        abs(ddG - exp_ddG) < expected_error
    ), f"Predicted ddG: {ddG}, expected value: {exp_ddG}, acceptable error: {expected_error}"
