from actpatt_dag.src.save import save_data


class Stats:

    def __init__(self):
        self.stats = {}

    def add(self, tag, value, time=None):

        if time is None:
            self.stats[tag] = value
        else:
            self.stats.setdefault(tag, []).append([time, value])

    def get(self, tag):

        return self.stats[tag]

    def save(self, file_name):
        save_data(self.stats, file_name)
