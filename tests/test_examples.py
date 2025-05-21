import matplotlib
matplotlib.use('Agg')

def test_import_example_modules():
    import phase_map_demo
    import por_deltae_grv_collector
    import secl_qa_cycle
    import history_evaluator


def test_scripts_run(tmp_path):
    import phase_map_demo
    import por_deltae_grv_collector
    import secl_qa_cycle

    # phase_map_demo main should run without errors
    phase_map_demo.main()

    # run a short cycle in the collector using the CLI in auto mode
    por_deltae_grv_collector.main([
        "--auto",
        "-n",
        "1",
        "-o",
        str(tmp_path / "out.csv"),
    ])

    # run a single step of the QA cycle
    secl_qa_cycle.main_qa_cycle(1, tmp_path / "hist.csv")

