import os
from pathlib import Path
import pytest
from hyperon import MeTTa, E


@pytest.fixture()
def metta():
    return MeTTa()


def pytest_collect_file(parent, file_path: Path):
    file_name, file_ext = os.path.splitext(file_path.name)
    if file_ext == ".metta" and file_name.startswith("test_"):
        return MeTTaFile.from_parent(parent, path=file_path)


def run_metta_test(metta, test_expr):
    result = metta.run(f"! ({test_expr})")[0][0]
    return result and result == E(), result


class MeTTaFile(pytest.File):
    def collect(self):
        metta = MeTTa()

        metta.run(self.path.read_text())

        # Read file and find (Test "name" function) forms
        tests = metta.run("! (match &self (Test $test-function) $test-function)")[0]

        for test in tests:
            test_name = test.get_name()
            yield MeTTaTest.from_parent(
                self, name=test_name, test_function=test_name, metta=metta
            )


class MeTTaTest(pytest.Item):
    def __init__(self, name, parent, test_function, metta):
        super().__init__(name, parent)
        self.test_function = test_function
        self.metta = metta

    def runtest(self):
        # Run MeTTa interpreter with self.source and evaluate self.test_function
        # This requires MeTTa interpreter bindings or a subprocess call
        try:
            passed, result = run_metta_test(self.metta, self.test_function)

            if not passed:
                raise MeTTaTestFailure(result)
        except Exception as e:
            raise MeTTaTestFailure(e)

    def repr_failure(self, excinfo):
        if isinstance(excinfo.value, MeTTaTestFailure):
            return str(excinfo.value)
        return super().repr_failure(excinfo)

    def reportinfo(self):
        return self.path, None, self.name


class MeTTaTestFailure(Exception):
    pass
