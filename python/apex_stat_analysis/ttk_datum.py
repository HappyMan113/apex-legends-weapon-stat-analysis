class TTKDatum:
    def __init__(self, clip_name: str, time_to_kill_secs: float):
        self.clip_name = clip_name
        self.time_to_kill_secs = time_to_kill_secs

    def get_clip_name(self):
        return self.clip_name

    def __float__(self):
        return self.time_to_kill_secs

    def __repr__(self):
        return f'Clip name: {self.clip_name}: {self.time_to_kill_secs:.2f} seconds'
