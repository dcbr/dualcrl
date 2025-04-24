import importlib
import pathlib


class Job:

    def __init__(self, job_name, uid):
        self._name = job_name
        self._uid = uid
        # Try importing the job file
        try:
            self._module = importlib.import_module(f"jobs.{job_name}")
            self._path = pathlib.Path("runs", job_name, f"{uid}")
        except ModuleNotFoundError:
            raise ValueError(f"Job not found: {job_name}")

    @property
    def name(self):
        return self._name

    @property
    def uid(self):
        return self._uid

    @property
    def path(self):
        return self._path

    def run_params(self, run_name=None):
        run_params = {params.name: params for params in self._module.setup_runs()}
        if run_name is None:
            return run_params
        else:
            if run_name in run_params:
                return run_params[run_name]
            else:
                raise ValueError(f"Run not found: {run_name}")

    def process_run(self, run, metrics, stats):
        return self._module.process_run(run, metrics, stats)

    def analyze_runs(self, job_path, runs):
        return self._module.analyze_runs(job_path, runs)
