import shutil
import os
import pytest


curr_dir = os.path.dirname(os.path.abspath(__file__))


@pytest.mark.parametrize(
    "jobname", ["ejm31_ejm45", "ejm31_ejm49", "ejm42_ejm48", "ejm44_ejm42"]
)
def _test_regression(tmp_path, jobname):
    from atm.rbfe_structprep import rbfe_structprep
    from atm.rbfe_production import rbfe_production
    from atm.uwham import calculate_uwham

    ref_ddG = {
        "ejm31_ejm45": -0.03,
        "ejm31_ejm49": 0.83,  # exp_ddG = 1.79
        "ejm42_ejm48": 0.68,  # exp_ddG = 0.78
        "ejm44_ejm42": -2.42,
    }[jobname]
    expected_error = {
        "ejm31_ejm45": 1.00,
        "ejm31_ejm49": 1.00,
        "ejm42_ejm48": 1.00,
        "ejm44_ejm42": 1.00,
    }[jobname]

    shutil.copytree(os.path.join(curr_dir, jobname), os.path.join(tmp_path, jobname))
    rbfe_structprep(os.path.join(tmp_path, jobname, f"{jobname}_input.yaml"))
    rbfe_production(os.path.join(tmp_path, jobname, f"{jobname}_input.yaml"))

    ddG = calculate_uwham(os.path.join(tmp_path, jobname), jobname, 100)[0]

    assert (
        abs(ddG - ref_ddG) < expected_error
    ), f"Predicted ddG: {ddG}, expected value: {ref_ddG}, acceptable error: {expected_error}"
