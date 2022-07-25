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
    #if "qiskit_token" in metafunc.fixturenames:
    #    print("enters")
    #    token = metafunc.config.getoption("qiskit_token")
    #    print(token)
    #    IBMQ.save_account(token)
    #provider = IBMQ.load_account()
    #backend = provider.get_backend('ibmq_qasm_simulator')