from src.utils.tracker import Tracker


class ResultAnalyzer:
    def __init__(self, params: dict):
        pass

    @staticmethod
    def clean_and_analyze_results(tracker: Tracker) -> dict:
        return tracker.to_dict()
