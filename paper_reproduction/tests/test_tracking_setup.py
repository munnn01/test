def test_tracking_local_environment_defines_both_ltr_and_tracker_settings(tmp_path):
    from reproduce import assets
    assert hasattr(assets,'write_tracking_environment'),'LTR legacy loader environment bootstrap is missing'
    source=tmp_path/'source';assets.write_tracking_environment(source,tmp_path/'assets')
    ltr={};exec((source/'ltr/admin/local.py').read_text(),ltr)
    settings=ltr['EnvironmentSettings']()
    assert settings.workspace_dir==str(tmp_path/'assets')
    assert (source/'pytracking/evaluation/local.py').is_file()
