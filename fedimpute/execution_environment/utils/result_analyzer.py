from fedimpute.execution_environment.utils.tracker import Tracker


class ResultAnalyzer:
    def __init__(self):
        pass

    @staticmethod
    def clean_and_analyze_results(tracker: Tracker) -> dict:
        return tracker.to_dict()
