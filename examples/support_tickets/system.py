from eval_harness.runner.system import SystemUnderEvaluation

class MockSystem(SystemUnderEvaluation):
    def run(self, case):
        return None

def create_system(**cfg):
    return MockSystem()
