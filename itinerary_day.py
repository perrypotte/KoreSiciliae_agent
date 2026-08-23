from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ItineraryDay:
    day: int
    starting_place = None
    ending_place = None
    selected_places: list = field(default_factory=list)
    remaining_time: Optional[float] = None

    def add_place(self, place):
        self.selected_places.append(place)