from dataclasses import dataclass, field
from itinerary_day import ItineraryDay
from typing import Optional
from geocoding import geocode_address
import json

@dataclass
class PlanningState:
    planning_mode: bool = False
    number_of_days: Optional[int] = None
    transport: Optional[str] = None
    max_distance_km: Optional[float] = None
    daily_hours: Optional[float] = None
    current_day: Optional[int] = None
    itinerary_days: list[ItineraryDay] = field(default_factory=list)

    def get_current_day(self):
        for day in self.itinerary_days:
            if day.day==self.current_day:
                temp=day
        return temp

    def set_current_day(self,current_day_itinerary):
        for day in self.itinerary_days:
                if day.day==current_day_itinerary.day:
                    day=current_day_itinerary

    def get_selected_places_ids(self):
        ids_list=list()
        for day in self.itinerary_days:
            for place in day.selected_places:
                ids_list.append(place["document_id"]) #MEMO Qui non è un oggettoma un dict normale quindi il punto non funziona
        return ids_list


    def add_place(self, place):
        self.get_current_day().add_place(place)

    def is_started(self):
        return len(self.itinerary_days)>0

    def add_new_day(self,starting_place,ending_place):
        self.itinerary_days.append(ItineraryDay(self.current_day,remaining_time=self.daily_hours))
        self.set_starting_ending_place(starting_place,ending_place)

    #TODO spostare metodi relativi a itineraryDay
    def set_starting_ending_place(self,starting_place,ending_place):
        for day in self.itinerary_days:
            if day.day==self.current_day:
                day.starting_place= geocode_address(starting_place) if starting_place!=None else geocode_address("Enna"),
                day.ending_place= geocode_address(ending_place) if ending_place!=None else geocode_address("Enna")

    def next_day(self):
        self.current_day += 1

    def to_prompt(self):
        state={
            "planning_mode": self.planning_mode,
            "number_of_days": self.number_of_days,
            "transport": self.transport,
            "max_distance_km": self.max_distance_km,
            "daily_hours": self.daily_hours,
            "current_day": self.current_day,
            "itinerary_days": [
                {
                    "day": day.day,
                    "starting_place": day.starting_place,
                    "ending_place": day.ending_place,
                    "selected_places": day.selected_places,
                    "remaining_time": day.remaining_time
                }
                for day in self.itinerary_days
            ]
        }
        return json.dumps(
                    state,
                    indent=2,
                    ensure_ascii=False
                )