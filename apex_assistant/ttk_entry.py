class Engagement:
    def __init__(self, ttff_seconds: float, enemy_distance_meters: float):
        self._ttff_seconds = ttff_seconds
        self._enemy_distance_meters = enemy_distance_meters

    def get_ttff_seconds(self) -> float:
        return self._ttff_seconds

    def get_enemy_distance_meters(self) -> float:
        return self._enemy_distance_meters
