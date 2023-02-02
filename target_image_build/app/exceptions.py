import json


class BuildLogEmptyError(Exception):
    def __init__(self, *args):
        super().__init__(*args)

    def __repr__(self):
        return "Build Log is Empty. Please Build Image first"

    def __str__(self):
        return "Build Log is Empty. Please Build Image first"


class AutoPushError(Exception):
    def __init__(self, *args):
        self.message = args
        # super().__init__(*args)

    """
    def __str__(self):
        return json.dumps(self.message[0])
    """
