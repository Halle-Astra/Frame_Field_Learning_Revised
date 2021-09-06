import os
class Barrier:
    def __init__(self):
        print(f"FILE:{os.path.split(__file__)[-1]}, INFO: an empty object named Barrier has been created.")
        pass

    def wait(self):
        pass
    def abort(self):
        print(f"FILE:{os.path.split(__file__)[-1]}, INFO: deleting barrier myself ing......")