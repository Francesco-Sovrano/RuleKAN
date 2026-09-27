def test_rulekan_namespace_exports_core_api():
    import rulekan

    assert rulekan.SumProductKAN is not None
    assert rulekan.PowerRuleKAN is not None
    assert rulekan.__version__ == "0.1.0"
