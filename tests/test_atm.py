import shutil
import os


curr_dir = os.path.dirname(os.path.abspath(__file__))


def _test_structprep(tmp_path):
    from atm.rbfe_structprep import rbfe_structprep

    shutil.copytree(
        os.path.join(curr_dir, "QB_A08_A07"), os.path.join(tmp_path, "QB_A08_A07")
    )
    rbfe_structprep(os.path.join(tmp_path, "QB_A08_A07", "QB_A08_A07_asyncre.yaml"))


def _test_production(tmp_path):
    from atm.rbfe_production import rbfe_production

    shutil.copytree(
        os.path.join(curr_dir, "QB_A08_A07_completed"),
        os.path.join(tmp_path, "QB_A08_A07_completed"),
    )
    rbfe_production(
        os.path.join(tmp_path, "QB_A08_A07_completed", "QB_A08_A07_asyncre.yaml")
    )


def _test_uwham_analysis(tmp_path):
    from atm.uwham import calculate_uwham

    shutil.copytree(
        os.path.join(curr_dir, "TYK2_A02_A09"), os.path.join(tmp_path, "TYK2_A02_A09")
    )

    run_dir = os.path.join(tmp_path, "TYK2_A02_A09", "TYK2_A02_A09_r0_1")
    ddG, ddG_std, _, _, _ = calculate_uwham(run_dir, "QB_A02_A09", 70)

    expected_ddG = -1.0582613105156682
    expected_ddG_std = 0.1967262793670161
    assert abs(ddG - expected_ddG) < 1e-8
    assert abs(ddG_std - expected_ddG_std) < 1e-8

    run_dir = os.path.join(tmp_path, "TYK2_A02_A09", "TYK2_A02_A09_r0_2")
    ddG, ddG_std, _, _, _ = calculate_uwham(run_dir, "QB_A02_A09", 70)

    expected_ddG = -1.9492444343451716
    expected_ddG_std = 0.18548486855089807
    assert abs(ddG - expected_ddG) < 1e-8
    assert abs(ddG_std - expected_ddG_std) < 1e-8

    run_dir = os.path.join(tmp_path, "TYK2_A02_A09", "TYK2_A02_A09_r0_3")
    ddG, ddG_std, _, _, _ = calculate_uwham(run_dir, "QB_A02_A09", None)

    expected_ddG = -1.2625416487680994
    expected_ddG_std = 0.2068400471106031
    assert abs(ddG - expected_ddG) < 1e-8
    assert abs(ddG_std - expected_ddG_std) < 1e-8
