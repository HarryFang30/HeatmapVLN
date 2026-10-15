"""System2 sampling for self-training rollouts: off unless asked for, and then the generate call samples."""

from __future__ import annotations

from scripts.evaluation import rpc_model_server as server

DEPLOYED = {"max_new_tokens": 128, "do_sample": False, "use_cache": True, "return_dict_in_generate": True}


def test_greedy_generate_is_exactly_the_deployed_call_when_unset_or_zero():
    assert server.system2_sampling({}) is None
    assert server.system2_generate_kwargs({}) == DEPLOYED
    assert server.system2_generate_kwargs({server.SYSTEM2_SAMPLE_TEMPERATURE_ENV: "0"}) == DEPLOYED
    assert server.system2_generate_kwargs({server.SYSTEM2_SAMPLE_TEMPERATURE_ENV: " "}) == DEPLOYED


def test_a_positive_temperature_samples():
    env = {server.SYSTEM2_SAMPLE_TEMPERATURE_ENV: "0.7", server.SYSTEM2_SAMPLE_TOP_P_ENV: "0.9"}
    assert server.system2_sampling(env) == {"temperature": 0.7, "top_p": 0.9}
    assert server.system2_generate_kwargs(env) == {**DEPLOYED, "do_sample": True, "temperature": 0.7, "top_p": 0.9}
    assert server.system2_sampling({server.SYSTEM2_SAMPLE_TEMPERATURE_ENV: "1"})["top_p"] == 1.0


def test_both_system2_generate_calls_use_the_switch():
    text = open(server.__file__, encoding="utf-8").read()
    assert text.count("**system2_generate_kwargs(),") == 2
    assert text.count("do_sample=False") == 1  # the audit replay stays greedy
