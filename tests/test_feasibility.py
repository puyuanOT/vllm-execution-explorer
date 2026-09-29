from runner.feasibility import main


def test_feasibility_command_stops_without_a_cuda_vllm_stack():
    assert main([]) == 2
