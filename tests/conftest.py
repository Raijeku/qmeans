def pytest_addoption(parser):
    parser.addoption(
        "--qiskit_token",
        action="store",
        default="MY_API_TOKEN",
        help="API token to access IBM Quantum services",
    )

def pytest_generate_tests(metafunc):
    option_value = metafunc.config.option.qiskit_token
    if 'qiskit_token' in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("qiskit_token", [option_value])